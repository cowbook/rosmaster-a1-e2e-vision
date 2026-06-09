from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import Tensor, nn


@dataclass
class PreprocessConfig:
    image_width: int = 200
    image_height: int = 66
    crop_top_pixels: int = 0


def preprocess_rgb_image(image_rgb: np.ndarray, cfg: PreprocessConfig) -> Tensor:
    if cfg.crop_top_pixels > 0 and cfg.crop_top_pixels < image_rgb.shape[0]:
        image_rgb = image_rgb[cfg.crop_top_pixels :, :, :]
    tensor = torch.from_numpy(image_rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = torch.nn.functional.interpolate(
        tensor,
        size=(cfg.image_height, cfg.image_width),
        mode="bilinear",
        align_corners=False,
    )
    return tensor


class PilotNetDualHead(nn.Module):
    def __init__(self, image_height: int = 66, image_width: int = 200) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_height, image_width)
            feature_size = int(torch.flatten(self.features(dummy), 1).shape[1])

        self.shared_head = nn.Sequential(
            nn.Linear(feature_size, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 10),
            nn.ReLU(inplace=True),
        )
        self.steering_head = nn.Linear(10, 1)
        self.stop_head = nn.Linear(10, 1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.shared_head(x)
        steering = torch.tanh(self.steering_head(x))
        stop_logit = self.stop_head(x)
        return steering, stop_logit


def masked_steering_loss(
    steering_pred: Tensor,
    steering_target: Tensor,
    go_target: Tensor,
    base_loss: nn.Module,
) -> Tensor:
    go_mask = (go_target > 0.5).float()
    if torch.sum(go_mask) < 1.0:
        return torch.zeros((), device=steering_pred.device)
    losses = base_loss(steering_pred, steering_target)
    masked = losses * go_mask
    return masked.sum() / go_mask.sum()
