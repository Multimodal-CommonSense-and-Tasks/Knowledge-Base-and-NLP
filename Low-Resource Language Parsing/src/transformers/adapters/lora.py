# Code adapted from https://github.com/microsoft/LoRA/blob/main/loralib/layers.py.
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_utils import PretrainedConfig
from .composition import AdapterCompositionBlock, Fuse, Stack
from .configuration import LoRAConfig
from .layer import AdapterLayerBase
from .context import ForwardContext
from .modeling import BertFusion, AvgFusion, MLPFusion


class LoRA(nn.Module):
    def __init__(
        self,
        lora_A_shape,
        lora_B_shape,
        config: LoRAConfig,
        gating_heads: int = 1,
    ):
        super().__init__()
        self.r = config.r
        self.lora_alpha = config.alpha
        self.composition_mode = config.composition_mode
        self.attn_matrices = config.attn_matrices
        self.use_gating = config.use_gating
        # Optional dropout
        if config.dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=config.dropout)
        else:
            self.lora_dropout = lambda x: x

        # Actual trainable parameters
        if self.r > 1 and self.composition_mode == "scale":
            raise ValueError("Can only use composition_mode='scale' when r == 1.")
        if self.r > 0:
            if self.composition_mode == "add":
                self.lora_A = nn.Parameter(torch.zeros(lora_A_shape))
            self.lora_B = nn.Parameter(torch.zeros(lora_B_shape))
            self.scaling = self.lora_alpha / self.r

            if self.use_gating:
                self.gate = nn.Linear(lora_A_shape[-1], gating_heads)

            if config.init_weights == "lora":
                # initialize A the same way as the default for nn.Linear and B to zero
                if self.composition_mode == "add":
                    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
                if self.use_gating:
                    nn.init.normal_(self.gate.weight, std=0.02)
            elif config.init_weights == "bert":
                if self.composition_mode == "add":
                    nn.init.normal_(self.lora_A, std=0.02)
                nn.init.normal_(self.lora_B, std=0.02)
                if self.use_gating:
                    nn.init.normal_(self.gate.weight, std=0.02)
            elif config.init_weights == "ia3":
                if self.composition_mode == "add":
                    nn.init.ones_(self.lora_A)
                nn.init.ones_(self.lora_B)
                if self.use_gating:
                    nn.init.normal_(self.gate.weight, std=0.02)
            else:
                raise ValueError("Unknown init_weights type: {}".format(config.init_weights))

    def com(self, weights: torch.Tensor, added: torch.Tensor, scaling=None) -> torch.Tensor:
        """Performs the composition operation between existing and injected weights."""
        if scaling is None:
            scaling = self.scaling
        if self.composition_mode == "add":
            return weights + added * scaling
        elif self.composition_mode == "scale":
            return weights * (added * scaling)
        else:
            raise ValueError("Invalid composition mode.")

    def com_inv(self, weights: torch.Tensor, added: torch.Tensor) -> torch.Tensor:
        """Inverts the composition operation between existing and injected weights."""
        if self.composition_mode == "add":
            return weights - added * self.scaling
        elif self.composition_mode == "scale":
            return weights / (added * self.scaling)
        else:
            raise ValueError("Invalid composition mode.")


class LoRALayer(AdapterLayerBase):
    def __init__(self, location_key: str, config: PretrainedConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.location_key = location_key + "_lora"
        self.config = config
        self.loras = nn.ModuleDict(dict())
        self.adapter_fusion_layer = nn.ModuleDict(dict())

        self.merged = False

    def get_n_heads(self, lora: Union[LoRA, LoRAConfig]):
        return 1

    def _check_lora_location(self, config: LoRAConfig):
        return True

    def _get_lora_shapes(self, config: LoRAConfig):
        raise NotImplementedError()

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        self.layer_idx = layer_idx
        lora_config = self.config.adapters.match(
            adapter_name,
            config_type=LoRAConfig,
            layer_idx=self.layer_idx,
            location_key=self.location_key,
        )
        if lora_config is not None and self._check_lora_location(lora_config):
            lora = LoRA(
                *self._get_lora_shapes(lora_config),
                lora_config,
                gating_heads=self.get_n_heads(lora_config),
            )
            lora.train(self.training)
            self.loras[adapter_name] = lora
            return True

        return False

    def delete_adapter(self, adapter_name: str):
        if adapter_name in self.loras:
            del self.loras[adapter_name]

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        adapter_names = adapter_names if isinstance(adapter_names, list) else adapter_names.split(",")
        # if self.config.adapters.common_config_value(adapter_names, self.location_key):
        if all([name in self.loras for name in adapter_names]):
            fusion_config = self.config.adapters.get_fusion(adapter_names)
            additional_kwargs = dict(invert_attention_score = hasattr(self.config, 'invert_fusion_score') and self.config.invert_fusion_score)
            fusion_cls = BertFusion
            if hasattr(self.config, 'use_avg_fusion') and self.config.use_avg_fusion:
                fusion_cls = AvgFusion
            if hasattr(self.config, 'mlp_fusion_mode') and self.config.mlp_fusion_mode:
                fusion_cls = MLPFusion
                additional_kwargs = dict(mode=self.config.mlp_fusion_mode,
                                         num_adapters=len(adapter_names),
                                         mlp_depth=self.config.mlp_fusion_depth,
                                         )
            fusion = fusion_cls(
                fusion_config,
                self.config.hidden_size,
                self.config.attention_probs_dropout_prob,
                **additional_kwargs
            )
            fusion.train(self.training)  # make sure training mode is consistent
            self.adapter_fusion_layer[",".join(adapter_names)] = fusion
            # pass  # not applicable to lora

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        adapter_names = adapter_names if isinstance(adapter_names, str) else ",".join(adapter_names)
        if adapter_names in self.adapter_fusion_layer:
            del self.adapter_fusion_layer[adapter_names]
        # pass  # not applicable to lora

    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_fusion: bool):
        if unfreeze_adapters:
            for name in adapter_setup.flatten():
                if name in self.loras:
                    for param in self.loras[name].parameters():
                        param.requires_grad = True

        if unfreeze_fusion:
            if isinstance(adapter_setup, Fuse):
                if adapter_setup.name in self.adapter_fusion_layer:
                    for param in self.adapter_fusion_layer[adapter_setup.name].parameters():
                        param.requires_grad = True
            for sub_setup in adapter_setup:
                if isinstance(sub_setup, Fuse):
                    if sub_setup.name in self.adapter_fusion_layer:
                        for param in self.adapter_fusion_layer[sub_setup.name].parameters():
                            param.requires_grad = True

    def get_adapter(self, adapter_name: str) -> nn.Module:
        if adapter_name in self.loras:
            return self.loras[adapter_name]
        else:
            return None


class Linear(LoRALayer, nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        location_key: str,
        config: PretrainedConfig,
        attn_key: str = None,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs
    ):
        LoRALayer.__init__(self, location_key, config, in_features, out_features, **kwargs)

        self.attn_key = attn_key
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = torch.t(self.weight.data)

    def _check_lora_location(self, config: LoRAConfig):
        return self.attn_key is None or self.attn_key in config.attn_matrices

    def _get_lora_shapes(self, config: LoRAConfig):
        return (config.r, self.in_features), (self.out_features, config.r)

    def reset_adapter(self):
        def T(w):
            return torch.t(w) if self.fan_in_fan_out else w

        if self.merged:
            lora = self.loras[self.merged]
            # Make sure that the weights are not merged
            if lora.r > 0:
                if lora.composition_mode == "scale":
                    delta_w = T(lora.lora_B)
                else:
                    delta_w = T(lora.lora_B @ lora.lora_A)
                self.weight.data = lora.com_inv(self.weight.data, delta_w)
            self.merged = None

    def _compute_adapted_weight(self, lora, scaling=None):
        def T(w):
            return torch.t(w) if self.fan_in_fan_out else w

        weight = self.weight
        # Merge the weights and mark it
        if lora.r > 0:
            if lora.composition_mode == "scale":
                delta_w = T(lora.lora_B)
            else:
                delta_w = T(lora.lora_B @ lora.lora_A)
            weight = lora.com(weight, delta_w, scaling=scaling)

        return weight

    def merge_adapter(self, name: str):
        if name in self.loras:
            if self.merged == name:
                return  # already merged
            elif not self.merged:
                lora = self.loras[name]
                if lora.use_gating:
                    raise ValueError("Cannot merge LoRA layer with gating.")
                self.weight.data = self._compute_adapted_weight(lora)
                self.merged = name
            elif self.merged != name:
                raise ValueError("LoRALayer already has a merged LoRA module. Please reset it first.")


    def adapter_fusion(self, adapter_setup: Fuse, x, result):
        """
        Performs adapter fusion with the given adapters for the given input.
        """
        context = ForwardContext.get_context()

        up_list = []

        for adapter_block in adapter_setup:
            # Case 1: We have a nested stack -> call stack method
            if isinstance(adapter_block, Stack):
                raise NotImplementedError
            # Case 2: We have a single adapter which is part of this module -> forward pass
            elif adapter_block in self.loras:
                up_list.append(self.forward_one_lora(adapter_block, x, result))
            # Case 3: nesting other composition blocks is invalid
            elif isinstance(adapter_block, AdapterCompositionBlock):
                raise ValueError(
                    "Invalid adapter setup. Cannot nest {} in {}".format(
                        adapter_block.__class__.__name__, adapter_setup.__class__.__name__
                    )
                )
            # Case X: No adapter which is part of this module -> ignore

        if len(up_list) > 0:
            orig_shape = up_list[0].shape
            up_list = torch.stack(up_list)
            assert len(orig_shape) == 3
            up_list = up_list.permute(1, 2, 0, 3) # TODO: check if this is correct

            fusion_output = self.adapter_fusion_layer[adapter_setup.name](
                result,
                up_list,
                up_list,
                residual=None,
                output_attentions=context.output_adapter_fusion_attentions,
            )
            if context.output_adapter_fusion_attentions:
                hidden_states = fusion_output[0]
                self._store_fusion_attentions(adapter_setup.name, fusion_output[-1])
            else:
                hidden_states = fusion_output

        return hidden_states

    def forward_one_lora(self, adapter, x, result):
        # result shape: <batch_size> x <seq_len> x <head_dim>
        lora = self.loras[adapter]
        if lora.r > 0:
            if lora.composition_mode == "scale":
                delta_w = lora.lora_B.view(1, 1, -1)
            else:
                delta_w = lora.lora_dropout(x) @ torch.t(lora.lora_A) @ torch.t(lora.lora_B)
            if lora.use_gating:
                gate = torch.sigmoid(lora.gate(x))
                gate = torch.mean(gate, dim=1).unsqueeze(-1)
                self._store_gating_score(adapter, gate)
            else:
                gate = None
            result = lora.com(result, delta_w, scaling=gate)
            return result
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor):
        def T(w):
            return torch.transpose(w, -2, -1) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)
        if not self.merged:
            adapter_setup = self.get_active_setup(self.loras)
            if adapter_setup is not None:
                for adapter in adapter_setup:
                    if isinstance(adapter, Fuse):
                        use_lora = [a in self.loras for a in adapter]
                        # assert all true or all false
                        all_lora = all([a for a in use_lora])
                        all_not_lora = not any([a for a in use_lora])
                        assert all_lora or all_not_lora
                        if all_lora:
                            result = self.adapter_fusion(adapter, x, result)
                    else:
                        if adapter in self.loras: # cares about unipelt + adapters
                            result = self.forward_one_lora(adapter, x, result)
                return result

        return result


class MergedLinear(LoRALayer, nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        location_key: str,
        config: PretrainedConfig,
        fan_in_fan_out: bool = False,
        **kwargs
    ):
        LoRALayer.__init__(self, location_key, config, in_features, out_features, **kwargs)

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def get_n_heads(self, lora: Union[LoRA, LoRAConfig]):
        return len(set(lora.attn_matrices))

    def _get_lora_shapes(self, config: LoRAConfig):
        n_heads = self.get_n_heads(config)
        return (config.r * n_heads, self.in_features), (
            self.out_features // 3 * n_heads,
            config.r,
        )

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        is_added = super().add_adapter(adapter_name, layer_idx)
        if is_added:
            lora_config = lora_config = self.config.adapters.match(
                adapter_name,
                config_type=LoRAConfig,
                layer_idx=self.layer_idx,
                location_key=self.location_key,
            )
            lora = self.loras[adapter_name]
            lora.enable_lora = [
                "q" in lora_config.attn_matrices,
                "k" in lora_config.attn_matrices,
                "v" in lora_config.attn_matrices,
            ]
            # Actual trainable parameters
            if any(lora.enable_lora):
                # Compute the indices
                lora.lora_ind = self.weight.new_zeros((self.out_features,), dtype=torch.bool).view(
                    len(lora.enable_lora), -1
                )
                lora.lora_ind[lora.enable_lora, :] = True
                lora.lora_ind = lora.lora_ind.view(-1)

    def pad(self, x, lora, fill_value=None):
        if fill_value is None:
            if lora.composition_mode == "add":
                fill_value = 0
            else:
                fill_value = 1
        result = x.new_full((*x.shape[:-1], self.out_features), fill_value)
        result = result.view(-1, self.out_features)
        result[:, lora.lora_ind] = x.reshape(-1, self.out_features // 3 * self.get_n_heads(lora))
        return result.view((*x.shape[:-1], self.out_features))

    def reset_adapter(self):
        def T(w):
            return w if self.fan_in_fan_out else torch.t(w)

        if self.merged:
            lora = self.loras[self.merged]
            # Make sure that the weights are not merged
            if lora.r > 0 and any(lora.enable_lora):
                if lora.composition_mode == "scale":
                    delta_w = lora.lora_B
                else:
                    delta_w = F.conv1d(
                        lora.lora_A.data.unsqueeze(0), lora.lora_B.data.unsqueeze(-1), groups=sum(lora.enable_lora)
                    ).squeeze(0)
                # shape after transpose: <head_dim> x <head_dim * n_heads>
                delta_w = delta_w.transpose(-2, -1)
                self.weight.data = lora.com_inv(self.weight.data, T(self.pad(delta_w, lora)))
            self.merged = None

    def _compute_adapted_weight(self, name, lora):
        def T(w):
            return w if self.fan_in_fan_out else torch.t(w)

        weight = self.weight
        if lora.r > 0:
            if lora.composition_mode == "scale":
                delta_w = lora.lora_B
            else:
                delta_w = F.conv1d(
                    lora.lora_A.data.unsqueeze(0), lora.lora_B.data.unsqueeze(-1), groups=sum(lora.enable_lora)
                ).squeeze(0)
            # shape after transpose: <head_dim> x <head_dim * n_heads>
            delta_w = delta_w.transpose(-2, -1)
            weight = lora.com(weight, T(self.pad(delta_w, lora)))

        return weight

    def merge_adapter(self, name: str):
        if name in self.loras:
            if self.merged == name:
                return  # already merged
            elif not self.merged:
                lora = self.loras[name]
                if lora.use_gating:
                    raise ValueError("Cannot merge LoRA layer with gating.")
                self.weight.data = self._compute_adapted_weight(name, lora)
                self.merged = name
            elif self.merged != name:
                raise ValueError("LoRALayer already has a merged LoRA module. Please reset it first.")

    def forward(self, x: torch.Tensor):
        def T(w):
            return torch.t(w) if self.fan_in_fan_out else w

        if not self.merged:
            adapter_setup = self.get_active_setup(self.loras)
            if adapter_setup is not None:
                if len(adapter_setup) == 1:
                    result = F.linear(x, T(self.weight), bias=self.bias)
                    lora = self.loras[adapter_setup[0]]
                    if lora.r > 0:
                        if lora.composition_mode == "scale":
                            delta_w = lora.lora_B.view(1, 1, -1)
                        else:
                            after_A = F.linear(lora.lora_dropout(x), lora.lora_A)
                            after_B = F.conv1d(
                                after_A.transpose(-2, -1), lora.lora_B.unsqueeze(-1), groups=sum(lora.enable_lora)
                            ).transpose(-2, -1)
                            delta_w = after_B
                        if lora.use_gating:
                            gate = torch.sigmoid(lora.gate(x))
                            gate = torch.mean(gate, dim=1)
                            self._store_gating_score(adapter_setup[0], gate)
                            gate = self.pad(
                                gate.repeat_interleave(self.out_features // 3, dim=-1), lora, fill_value=1
                            ).unsqueeze(1)
                        else:
                            gate = None
                        # result = (batch_size, seq_len, head_dim * 3)
                        result = lora.com(result, self.pad(delta_w, lora), scaling=gate)
                    return result
                else:
                    raise ValueError(f"Invalid adapter setup. Cannot use {adapter_setup} with LoRA.")

        return F.linear(x, T(self.weight), bias=self.bias)
