from typing import List

import torch
from torch.nn import ParameterList, Parameter

from allennlp.common.checks import ConfigurationError

class ScalarMixWithDropout(torch.nn.Module):
    """
    Computes a parameterised scalar mixture of N tensors, ``mixture = gamma * sum(s_k * tensor_k)``
    where ``s = softmax(w)``, with ``w`` and ``gamma`` scalar parameters.

    Applies Layer Dropout (Kondratyuk and Straka, 2019) to the weights ``w`` prior to computing
    the scalar mix.

    In addition, if ``do_layer_norm=True`` then apply layer normalization to each tensor
    before weighting.
    """
    def __init__(self,
                 mixture_size: int,
                 layer_dropout: float = 0.0,
                 layer_dropout_replacement: float = -1e20,
                 do_layer_norm: bool = False,
                 initial_scalar_parameters: List[float] = None,
                 trainable: bool = True) -> None:
        super(ScalarMixWithDropout, self).__init__()
        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm
        self.layer_dropout = layer_dropout

        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size
        elif len(initial_scalar_parameters) != mixture_size:
            raise ConfigurationError("Length of initial_scalar_parameters {} differs "
                                     "from mixture_size {}".format(
                                             initial_scalar_parameters, mixture_size))

        self.scalar_parameters = ParameterList(
                [Parameter(torch.FloatTensor([initial_scalar_parameters[i]]),
                           requires_grad=trainable) for i
                 in range(mixture_size)])
        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)
        
        if self.layer_dropout:
            # largely adapted from
            # https://github.com/Hyperparticle/udify/blob/master/udify/modules/scalar_mix.py
            # we can't use nn.Dropout(*) here because it sets things to 0,
            # so when softmax the weights, we still assign a non-zero weight
            # to the layer.  instead, we need to set the param to -\inf so that
            # exp(*) = 0
            dropout_mask = torch.zeros(len(self.scalar_parameters))
            dropout_fill = torch.empty(len(self.scalar_parameters)).fill_(layer_dropout_replacement)
            self.register_buffer("dropout_mask", dropout_mask)
            self.register_buffer("dropout_fill", dropout_fill)

    def forward(self, tensors: List[torch.Tensor],  # pylint: disable=arguments-differ
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute a weighted average of the ``tensors``.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.

        When ``do_layer_norm=True``, the ``mask`` is required input.  If the ``tensors`` are
        dimensioned  ``(dim_0, ..., dim_{n-1}, dim_n)``, then the ``mask`` is dimensioned
        ``(dim_0, ..., dim_{n-1})``, as in the typical case with ``tensors`` of shape
        ``(batch_size, timesteps, dim)`` and ``mask`` of shape ``(batch_size, timesteps)``.

        When ``do_layer_norm=False`` the ``mask`` is ignored.
        """
        if len(tensors) != self.mixture_size:
            raise ConfigurationError("{} tensors were passed, but the module was initialized to "
                                     "mix {} tensors.".format(len(tensors), self.mixture_size))

        def _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked):
            tensor_masked = tensor * broadcast_mask
            mean = torch.sum(tensor_masked) / num_elements_not_masked
            variance = torch.sum(((tensor_masked - mean) * broadcast_mask)**2) / num_elements_not_masked
            return (tensor - mean) / torch.sqrt(variance + 1E-12)

        # layer dropout
        weights = torch.cat([parameter for parameter in self.scalar_parameters])
        if self.layer_dropout:
            weights = torch.where(self.dropout_mask.uniform_() > self.layer_dropout,
                                   weights,
                                   self.dropout_fill)

        normed_weights = torch.nn.functional.softmax(weights, dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        if not self.do_layer_norm:
            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * tensor)
            return self.gamma * sum(pieces)

        else:
            mask_float = mask.float()
            broadcast_mask = mask_float.unsqueeze(-1)
            input_dim = tensors[0].size(-1)
            num_elements_not_masked = torch.sum(mask_float) * input_dim

            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * _do_layer_norm(tensor,
                                                      broadcast_mask, num_elements_not_masked))
            return self.gamma * sum(pieces)