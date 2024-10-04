from typing import List, Union

import torch
from torch import nn

from ..modeling_utils import ModuleUtilsMixin
from .composition import AdapterCompositionBlock, Fuse, Stack
from .configuration import PrefixTuningConfig
from .context import AdapterSetup, ForwardContext
from .layer import AdapterLayerBase
from .modeling import Activation_Function_Class
from .modeling import BertFusion, AvgFusion, MLPFusion


class PrefixTuning(nn.Module, ModuleUtilsMixin):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        input_size: int,
        config: PrefixTuningConfig,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.input_size = input_size
        self.n_embd_per_head = self.input_size // self.n_heads
        self.config = config

        self.wte = nn.Embedding(self.config.prefix_length, self.input_size)
        self.control_trans = nn.Sequential(
            nn.Linear(self.input_size, self.config.bottleneck_size),
            Activation_Function_Class(self.config.non_linearity.lower()),
            nn.Linear(self.config.bottleneck_size, self.n_layers * 2 * self.input_size),
        )
        self.dropout = nn.Dropout(self.config.dropout)

    def eject(self):
        input_tokens = torch.arange(self.config.prefix_length).long()
        input_tokens = input_tokens.unsqueeze(0).expand(1, -1).to(self.device)
        embs = self.wte(input_tokens)
        key_values = self.control_trans(embs)  # batch_size x prefix_length x n_layers*2*input_size
        key_values = key_values.view(
            self.config.prefix_length * self.n_layers * 2 * self.input_size
        )  # *2 for key and value

        return key_values

    def forward(self, batch_size):
        input_tokens = torch.arange(self.config.prefix_length).long()
        input_tokens = input_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)
        embs = self.wte(input_tokens)
        key_values = self.control_trans(embs)  # batch_size x prefix_length x n_layers*2*input_size
        key_values = key_values.view(
            batch_size, self.config.prefix_length, self.n_layers * 2, self.n_heads, self.n_embd_per_head
        )  # *2 for key and value
        key_values = self.dropout(key_values)
        # n_layers * (2 x batch_size x n_heads x prefix_length x n_embd_per_head)
        key_values = key_values.permute(2, 0, 3, 1, 4).split(2)

        return key_values


class FlatPrefixTuning(nn.Module, ModuleUtilsMixin):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        input_size: int,
        config: PrefixTuningConfig,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.input_size = input_size
        self.n_embd_per_head = self.input_size // self.n_heads
        self.config = config

        self.control_trans = nn.Parameter(torch.randn(self.config.prefix_length * self.n_layers * 2 * self.input_size))

        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, batch_size):
        key_values = (
            self.control_trans.unsqueeze(0)
            .expand(batch_size, -1)
            .view(batch_size, self.config.prefix_length, self.n_layers * 2, self.n_heads, self.n_embd_per_head)
            .to(self.device)
        )  # *2 for key and value
        key_values = self.dropout(key_values)
        # n_layers * (2 x batch_size x n_heads x prefix_length x n_embd_per_head)
        key_values = key_values.permute(2, 0, 3, 1, 4).split(2)

        return key_values


class PrefixTuningGroup(nn.ModuleDict):
    def __init__(self, module_configs, prefix_tuning_config):
        super().__init__()
        if prefix_tuning_config["flat"]:
            prefix_tuning_class = FlatPrefixTuning
        else:
            prefix_tuning_class = PrefixTuning
        for k, kwargs in module_configs.items():
            self[k] = prefix_tuning_class(**kwargs, config=prefix_tuning_config)

    def eject(self):
        """Converts all PrefixTuning modules into FlatPrefixTuning modules."""
        for k, v in self.items():
            if isinstance(v, PrefixTuning):
                config = v.config.replace(flat=True)
                self[k] = FlatPrefixTuning(v.n_layers, v.n_heads, v.input_size, config)
                weights = v.eject()
                self[k].control_trans = nn.Parameter(weights)

    def forward(self, batch_size):
        return {k: v(batch_size) for k, v in self.items()}


class PrefixTuningPool(nn.Module):
    """
    The model layer that holds all Prefix Tuning prefixes. While each Transformers layer has its own prefix, this layer
    is shared across all Transformers layers.

    How it works:

        1. A `PrefixTuningShim` module that sets this module as pool module is added to each layer.
        2. On adding a prefix, each shim module where a prefix should be added increments a counter in `prefix_counts`.
        3. Finally, the base model class confirms adding a new prefix by calling `confirm_prefix()`.
        4. This module adds a prefix layer that produces outputs corresponding to the indicated number of layers.

    Notes:

        - The forward call to this layer is executed in the ForwardContext of each model pass.
        - All other methods of this class (except for `confirm_prefix()`) should be called exclusively by
          `PrefixTuningShim`.

    Args:
        config (:class:`~transformers.PretrainedConfig`): The model config.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.prefix_counts = {}
        self.prefix_tunings = nn.ModuleDict()

    def indicate_prefix(self, prefix_name: str, location_key: str):
        if prefix_name not in self.prefix_counts:
            self.prefix_counts[prefix_name] = {location_key: 1}
        elif location_key not in self.prefix_counts[prefix_name]:
            self.prefix_counts[prefix_name][location_key] = 1
        else:
            self.prefix_counts[prefix_name][location_key] += 1

        return self.prefix_counts[prefix_name][location_key] - 1

    def confirm_prefix(self, prefix_name: str):
        """Create Prefix Tuning module based on shim layer infications."""
        prefix_tuning_config = self.config.adapters.match(prefix_name, PrefixTuningConfig)
        if prefix_tuning_config is None:
            return

        if prefix_name not in self.prefix_counts:
            raise ValueError(f"Prefix {prefix_name} not found in PrefixTuningPool")

        module_configs = {}
        for location_key, count in self.prefix_counts[prefix_name].items():
            module_configs[location_key] = {
                "n_layers": count,
                "n_heads": self.config.num_attention_heads,
                "input_size": self.config.hidden_size,
            }
        prefix_tuning = PrefixTuningGroup(module_configs, prefix_tuning_config)
        prefix_tuning.train(self.training)  # make sure training mode is consistent
        self.prefix_tunings[prefix_name] = prefix_tuning
        del self.prefix_counts[prefix_name]

    def delete_prefix(self, prefix_name: str):
        if prefix_name in self.prefix_tunings:
            del self.prefix_tunings[prefix_name]

    def enable_prefix(self, prefix_name: str):
        if prefix_name in self.prefix_tunings:
            for param in self.prefix_tunings[prefix_name].parameters():
                param.requires_grad = True

    def get_prefix(self, prefix_name: str):
        if prefix_name in self.prefix_tunings:
            return self.prefix_tunings[prefix_name]
        else:
            return None

    def forward(self, *args, **kwargs):
        context = AdapterSetup.get_context()
        if context is not None:
            adapter_setup = context.adapter_setup
        else:
            adapter_setup = self.config.adapters.active_setup

        prefix_states = {}
        if adapter_setup is not None:
            # Infer batch size
            input_tensor_names = ["input_ids", "decoder_input_ids", "attention_mask", "inputs_embeds", "pixel_values"]
            batch_size = None
            for name in input_tensor_names:
                if kwargs.get(name, None) is not None:
                    batch_size = kwargs[name].size(0)
                    break
            if batch_size is None:
                if len(args) > 0:
                    batch_size = args[0].size(0)
                else:
                    raise ValueError("Could not infer batch size for prefix tuning from inputs.")

            # Pass to sub-layers
            for name in adapter_setup.flatten():
                if name in self.prefix_tunings:
                    prefix_states[name] = self.prefix_tunings[name](batch_size)

        return prefix_states


class PrefixTuningShim(AdapterLayerBase, nn.Module):
    """
    Representation of a Prefix Tuning layer within one Transformer layer. This class implements `AdapterLayerBase` for
    compatibility with adapters. It uses `PrefixTuningPool` in the background and `set_pool()` must be called after
    initialization.

    Args:
        location_key (str): The id describing the location of this layer in the model.
                            Currently, can be "encoder_prefix", "cross_prefix" or None.
        config (:class:`~transformers.PretrainedConfig`): The model config.
    """

    def __init__(self, location_key: str, config):
        super().__init__()
        self.config = config
        self.location_key = location_key
        self.prefixes = {}
        self.prefix_gates = nn.ModuleDict()
        self.adapter_fusion_layer = nn.ModuleDict(dict())

    def set_pool(self, pool: PrefixTuningPool):
        self.__setattr__("pool", pool)

    def add_adapter(self, adapter_name: str, layer_idx: int):
        self.layer_idx = layer_idx
        # only match location keys for which we have config keys
        if self.location_key.startswith("cross") or self.location_key.startswith("encoder"):
            used_location_key = self.location_key
        else:
            used_location_key = None
        prefix_tuning_config = self.config.adapters.match(
            adapter_name,
            config_type=PrefixTuningConfig,
            layer_idx=self.layer_idx,
            location_key=used_location_key,
        )
        if prefix_tuning_config is not None:
            prefix_id = self.pool.indicate_prefix(adapter_name, self.location_key)
            self.prefixes[adapter_name] = prefix_id

            if prefix_tuning_config.use_gating:
                gate_outputs = 1 if prefix_tuning_config.shared_gating else 2
                gate = nn.Linear(self.config.hidden_size, gate_outputs)
                gate.weight.data.normal_(mean=0.0, std=0.02)
                self.prefix_gates[adapter_name] = gate

    def delete_adapter(self, adapter_name: str):
        self.pool.delete_prefix(adapter_name)
        if adapter_name in self.prefixes:
            del self.prefixes[adapter_name]
        if adapter_name in self.prefix_gates:
            del self.prefix_gates[adapter_name]

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        adapter_names = adapter_names if isinstance(adapter_names, list) else adapter_names.split(",")
        # if self.config.adapters.common_config_value(adapter_names, self.location_key):
        if all([name in self.prefixes for name in adapter_names]):
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

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        adapter_names = adapter_names if isinstance(adapter_names, str) else ",".join(adapter_names)
        if adapter_names in self.adapter_fusion_layer:
            del self.adapter_fusion_layer[adapter_names]
        # pass  # not applicable to prefix tuning

    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_fusion: bool):
        if unfreeze_adapters:
            for prefix_tuning_name in adapter_setup.flatten():
                self.pool.enable_prefix(prefix_tuning_name)
                if prefix_tuning_name in self.prefix_gates:
                    for param in self.prefix_gates[prefix_tuning_name].parameters():
                        param.requires_grad = unfreeze_adapters

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

    def get_adapter(self, adapter_name):
        return_dict = nn.ModuleDict()
        # Make sure to only return params once
        if adapter_name in self.prefixes and self.prefixes[adapter_name] == 0:
            prefix_module = self.pool.get_prefix(adapter_name)
            if prefix_module is not None:
                return_dict["prefix"] = prefix_module[self.location_key]
        if adapter_name in self.prefix_gates:
            return_dict["gate"] = self.prefix_gates[adapter_name]
        if len(return_dict) > 0:
            return return_dict

        return None

    def adapter_fusion(self, adapter_setup: Fuse, key_states, value_states, residual_input):
        """
        Performs adapter fusion with the given adapters for the given input.
        """
        context = ForwardContext.get_context()

        state_dict = {"key": key_states, "val": value_states}
        up_list_dict = {"key": [], "val": []}

        for adapter_block in adapter_setup:
            # Case 1: We have a nested stack -> call stack method
            if isinstance(adapter_block, Stack):
                raise NotImplementedError
            # Case 2: We have a single adapter which is part of this module -> forward pass
            elif adapter_block in self.prefixes:
                up_key, up_val, prefix_keys = self.forward_one_prefix(adapter_block, key_states, value_states, residual_input)
                up_list_dict['key'].append(up_key)
                up_list_dict['val'].append(up_val)

            # Case 3: nesting other composition blocks is invalid
            elif isinstance(adapter_block, AdapterCompositionBlock):
                raise ValueError(
                    "Invalid adapter setup. Cannot nest {} in {}".format(
                        adapter_block.__class__.__name__, adapter_setup.__class__.__name__
                    )
                )
            # Case X: No adapter which is part of this module -> ignore

        out_dict = {}
        for key in ['key', 'val']:
            up_list = up_list_dict[key]
            if len(up_list) > 0:
                orig_shape = up_list[0].shape
                up_list = torch.stack(up_list)
                if len(orig_shape) == 4: # up_list (num_adapters, batch, num_heads, seq_len, dim)
                    up_list = up_list.permute(1, 3, 0, 2, 4) # (batch, seq_len, num_adapters, num_heads, dim)
                    up_list = up_list.reshape(*up_list.shape[:3], -1) # (batch, seq_len, num_adapters, dim*num_heads)
                    state_dict[key] = state_dict[key].permute(0, 2, 1, 3)
                    state_dict[key] = state_dict[key].reshape(*state_dict[key].shape[:2], -1) # (batch, seq_len, dim*num_heads)
                else:
                    assert len(orig_shape) == 3 # up_list (num_adapters, batch, seq_len, dim)
                    up_list = up_list.permute(1, 2, 0, 3)

                fusion_output = self.adapter_fusion_layer[adapter_setup.name](
                    state_dict[key],
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

                if len(orig_shape) == 4:
                    batch, num_heads, seq_len, dim = orig_shape
                    hidden_states = hidden_states.reshape(batch, seq_len, num_heads, dim)
                    hidden_states = hidden_states.permute(0, 2, 1, 3) # (batch, num_heads, seq_len, dim)

            out_dict[key] = hidden_states

        return out_dict['key'], out_dict['val'], prefix_keys

    def forward_one_prefix(self, prefix_tuning_name, key_states, value_states, residual_input):
        prefix_id = self.prefixes[prefix_tuning_name]

        # Retrieve pre-computed prefix states from context
        context = ForwardContext.get_context()
        # batch_size x n_heads x prefix_length x n_embd_per_head
        prefix_keys, prefix_values = context.prefix_states[prefix_tuning_name][self.location_key][
            prefix_id
        ]

        if prefix_tuning_name in self.prefix_gates:
            gate = self.prefix_gates[prefix_tuning_name]
            gate_output = torch.mean(torch.sigmoid(gate(residual_input)), dim=1)
            self._store_gating_score(prefix_tuning_name, gate_output)
            gate_output_key = gate_output[:, 0].view(-1, 1, 1, 1)
            gate_output_value = gate_output[:, -1].view(-1, 1, 1, 1)
            key_states = key_states * gate_output_key
            value_states = value_states * gate_output_value

        key_states = torch.cat([prefix_keys, key_states], dim=2)
        value_states = torch.cat([prefix_values, value_states], dim=2)
        return key_states, value_states, prefix_keys

    def forward(self, key_states, value_states, residual_input, attention_mask=None, invert_mask=True):
        adapter_setup = self.get_active_setup(self.prefixes)
        if adapter_setup is not None:
            for prefix_tuning_name in adapter_setup:
                batch_size = key_states.size(0)
                ran_prefix_tuning = False

                if isinstance(prefix_tuning_name, Fuse):
                    use_prefix = [a in self.prefixes for a in prefix_tuning_name]
                    # assert all true or all false
                    all_prefix = all([a for a in use_prefix])
                    all_not_prefix = not any([a for a in use_prefix])
                    assert all_prefix or all_not_prefix
                    if all_prefix:
                        key_states, value_states, prefix_keys = self.adapter_fusion(prefix_tuning_name, key_states, value_states, residual_input)
                        ran_prefix_tuning = True
                else:
                    # if prefix_tuning_name in self.prefixes:
                    if prefix_tuning_name in self.prefixes:  # cares about unipelt + adapters
                        key_states, value_states, prefix_keys = self.forward_one_prefix(prefix_tuning_name, key_states, value_states, residual_input)
                        ran_prefix_tuning = True

                if ran_prefix_tuning and attention_mask is not None:
                    if attention_mask.dim() == 2:
                        prefix_mask = torch.ones(batch_size, prefix_keys.size(2)).to(attention_mask.device)
                    else:
                        prefix_mask = torch.ones(batch_size, 1, attention_mask.size(2), prefix_keys.size(2)).to(
                            attention_mask.device
                        )
                    if invert_mask:
                        prefix_mask = 1.0 - prefix_mask
                    attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)
            # else:
            #     raise ValueError(f"Invalid adapter setup. Cannot use {adapter_setup} with prefix tuning.")

        return key_states, value_states, attention_mask
