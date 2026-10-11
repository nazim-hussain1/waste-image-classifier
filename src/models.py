import torch.nn as nn
from torchvision import models


def small_cnn(n):
    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n))


def build_model(name, n_classes, freeze_backbone=False):
    if name == "baseline":
        return small_cnn(n_classes)
    if name == "mobilenet":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        if freeze_backbone:
            for p in m.features.parameters():
                p.requires_grad = False
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
        return m
    raise ValueError(f"Unknown model: {name}")
