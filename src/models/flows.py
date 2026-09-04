"""Masked autoregressive flows for task-conditional feature density."""

from __future__ import annotations

import math

import torch
from torch import nn


class MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor):
        super().__init__(in_features, out_features)
        self.register_buffer("mask", mask)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(inputs, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_multiplier: int = 2,
    ) -> None:
        super().__init__()
        degrees = _build_degrees(input_dim, hidden_dims, output_multiplier)
        masks = _build_masks(degrees)
        layer_dims = [input_dim, *hidden_dims, input_dim * output_multiplier]
        layers: list[nn.Module] = []
        for layer_index, (in_dim, out_dim) in enumerate(
            zip(layer_dims[:-1], layer_dims[1:])
        ):
            layers.append(MaskedLinear(in_dim, out_dim, masks[layer_index]))
            if layer_index < len(layer_dims) - 2:
                layers.append(nn.GELU())
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class MaskedAutoregressiveLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        log_scale_clip: float,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.log_scale_clip = log_scale_clip
        self.net = MADE(input_dim=input_dim, hidden_dims=hidden_dims)

    def forward_transform(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale = self.net(inputs).chunk(2, dim=-1)
        log_scale = self.log_scale_clip * torch.tanh(log_scale / self.log_scale_clip)
        outputs = (inputs - shift) * torch.exp(-log_scale)
        return outputs, -log_scale.sum(dim=-1)


class ReversePermutation(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.register_buffer("permutation", torch.arange(input_dim - 1, -1, -1))

    def forward_transform(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = inputs[:, self.permutation]
        return outputs, torch.zeros(inputs.size(0), device=inputs.device)


class MaskedAutoregressiveFlow(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        hidden_dims: list[int],
        log_scale_clip: float = 3.0,
    ) -> None:
        super().__init__()
        if input_dim < 2:
            raise ValueError("A masked autoregressive flow requires input_dim >= 2.")
        if num_layers < 1:
            raise ValueError("num_layers must be positive.")
        transforms: list[nn.Module] = []
        for layer_index in range(num_layers):
            transforms.append(
                MaskedAutoregressiveLayer(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    log_scale_clip=log_scale_clip,
                )
            )
            if layer_index < num_layers - 1:
                transforms.append(ReversePermutation(input_dim))
        self.transforms = nn.ModuleList(transforms)

    def log_prob(self, inputs: torch.Tensor) -> torch.Tensor:
        transformed = inputs
        total_log_det = torch.zeros(inputs.size(0), device=inputs.device)
        for transform in self.transforms:
            transformed, log_det = transform.forward_transform(transformed)
            total_log_det = total_log_det + log_det
        base_log_prob = -0.5 * (transformed.pow(2) + math.log(2.0 * math.pi)).sum(
            dim=-1
        )
        return base_log_prob + total_log_det


def build_small_maf(
    input_dim: int,
    num_layers: int,
    hidden_multiplier: int,
    hidden_layers: int,
    log_scale_clip: float,
) -> MaskedAutoregressiveFlow:
    if hidden_multiplier < 1 or hidden_layers < 1:
        raise ValueError("hidden_multiplier and hidden_layers must be positive.")
    hidden_dim = input_dim * hidden_multiplier
    return MaskedAutoregressiveFlow(
        input_dim=input_dim,
        num_layers=num_layers,
        hidden_dims=[hidden_dim] * hidden_layers,
        log_scale_clip=log_scale_clip,
    )


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _build_degrees(
    input_dim: int,
    hidden_dims: list[int],
    output_multiplier: int,
) -> list[torch.Tensor]:
    input_degrees = torch.arange(1, input_dim + 1)
    degrees = [input_degrees]
    for hidden_dim in hidden_dims:
        degrees.append((torch.arange(hidden_dim) % max(input_dim - 1, 1)) + 1)
    degrees.append(input_degrees.repeat(output_multiplier))
    return degrees


def _build_masks(degrees: list[torch.Tensor]) -> list[torch.Tensor]:
    masks: list[torch.Tensor] = []
    for layer_index in range(len(degrees) - 1):
        left = degrees[layer_index]
        right = degrees[layer_index + 1]
        if layer_index == len(degrees) - 2:
            mask = (right.unsqueeze(-1) > left.unsqueeze(0)).float()
        else:
            mask = (right.unsqueeze(-1) >= left.unsqueeze(0)).float()
        masks.append(mask)
    return masks
