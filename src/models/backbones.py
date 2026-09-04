"""ARC-equivalent ImageNet-pretrained feature extractors."""

from __future__ import annotations

from dataclasses import dataclass

ARC_RESNET50_URL = "https://download.pytorch.org/models/resnet50-19c8e357.pth"


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    feature_dim: int
    source: str


class FrozenBackbone:
    """A module wrapper that prevents parameter and BatchNorm state updates."""

    def __init__(self, module, spec: BackboneSpec) -> None:
        import torch

        if not isinstance(module, torch.nn.Module):
            raise TypeError("module must be a torch.nn.Module")
        self.module = module
        self.spec = spec
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.module.eval()

    def to(self, device):
        self.module.to(device)
        self.module.eval()
        return self

    def eval(self):
        self.module.eval()
        return self

    def __call__(self, images):
        import torch

        self.module.eval()
        with torch.no_grad():
            return self.module(images)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        return self.module.load_state_dict(state_dict)


def build_frozen_backbone(name: str, pretrained: bool = True) -> FrozenBackbone:
    if name == "vit_b16":
        import timm

        module = timm.create_model(
            "vit_base_patch16_224", pretrained=pretrained, num_classes=0
        )
        return FrozenBackbone(
            module,
            BackboneSpec(
                name=name,
                feature_dim=768,
                source="timm==0.6.12:vit_base_patch16_224 ImageNet-1K fine-tuned from ImageNet-21K",
            ),
        )

    if name == "resnet50":
        import torch
        from torchvision.models import resnet50

        module = resnet50(weights=None)
        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(
                ARC_RESNET50_URL, map_location="cpu", check_hash=True, progress=True
            )
            module.load_state_dict(state_dict, strict=True)
        module.fc = torch.nn.Identity()
        return FrozenBackbone(
            module,
            BackboneSpec(
                name=name,
                feature_dim=2048,
                source=f"ARC bundled torchvision ResNet-50 weights: {ARC_RESNET50_URL}",
            ),
        )
    raise KeyError(f"Unknown backbone: {name}")
