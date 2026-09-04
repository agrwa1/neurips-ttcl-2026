"""ARC/PILOT-compatible expanding linear classifier."""

from __future__ import annotations


def _torch():
    import torch

    return torch


class ExpandingLinearClassifier:
    """Own a torch Linear-like module and preserve rows when classes are appended.

    ARC's SimpleLinear initializes weights with Kaiming uniform using a linear
    nonlinearity and initializes biases to zero. Expansion initializes a new full
    matrix, then copies all old rows verbatim.
    """

    def __init__(self, feature_dim: int, device) -> None:
        self.feature_dim = int(feature_dim)
        self.device = device
        self.module = None

    @property
    def out_features(self) -> int:
        return 0 if self.module is None else int(self.module.out_features)

    def expand(self, num_classes: int):
        torch = _torch()
        if num_classes <= self.out_features:
            raise ValueError(
                "Classifier expansion must increase the number of classes."
            )
        new_module = torch.nn.Linear(self.feature_dim, int(num_classes), bias=True).to(
            self.device
        )
        torch.nn.init.kaiming_uniform_(new_module.weight, nonlinearity="linear")
        torch.nn.init.constant_(new_module.bias, 0.0)
        if self.module is not None:
            old_count = self.out_features
            with torch.no_grad():
                new_module.weight[:old_count].copy_(self.module.weight)
                new_module.bias[:old_count].copy_(self.module.bias)
        self.module = new_module
        return self.module

    def __call__(self, features):
        if self.module is None:
            raise RuntimeError("Classifier has not been initialized.")
        return self.module(features)

    def state_dict(self):
        if self.module is None:
            raise RuntimeError("Classifier has not been initialized.")
        return self.module.state_dict()
