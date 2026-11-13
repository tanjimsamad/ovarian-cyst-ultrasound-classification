"""Model definitions: the custom CNN and the ImageNet-pretrained ResNet18."""
import torch.nn as nn

from .config import CLASS_NAMES

_FILTERS = {
    "manuscript": [32, 32, 64, 64, 128],
    "wider":      [32, 64, 128, 256, 256],
    "deeper":     [32, 32, 64, 64, 128, 128],   # architecture of the reported CNN
    "gap":        [32, 64, 128, 256],
}


class TunableCNN(nn.Module):
    """Plain VGG-style CNN. Every block = Conv3x3 -> (BatchNorm) -> ReLU -> MaxPool2.

    `manuscript` has no BatchNorm and a flatten + 2 hidden FC head; all other variants use
    BatchNorm, global average pooling and a 128-unit hidden FC layer.
    """

    def __init__(self, architecture="deeper", num_classes=len(CLASS_NAMES), dropout=0.3):
        super().__init__()
        if architecture not in _FILTERS:
            raise ValueError(f"unknown architecture {architecture!r}")
        filters = _FILTERS[architecture]
        use_bn = use_gap = architecture != "manuscript"

        layers, c_in = [], 3
        for c_out in filters:
            layers.append(nn.Conv2d(c_in, c_out, 3, padding=1, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(c_out))
            layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)

        if use_gap:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(filters[-1], 128), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(128, num_classes))
        else:   # 224 -> 7 after five poolings
            self.pool = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(128 * 7 * 7, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(128, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_resnet18(num_classes=len(CLASS_NAMES), pretrained=True):
    """torchvision ResNet18; the 1000-way head is replaced by a `num_classes`-way linear layer.
    All layers are fine-tuned (nothing is frozen)."""
    from torchvision import models
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def build_model(name, **cfg):
    if name == "cnn":
        return TunableCNN(cfg.get("architecture", "deeper"), dropout=cfg.get("dropout", 0.3))
    if name == "resnet18":
        return build_resnet18(pretrained=cfg.get("pretrained", True))
    raise ValueError(name)
