from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rosmaster_a1_e2e_vision.model import (  # noqa: E402
    PilotNetDualHead,
    PreprocessConfig,
    masked_steering_loss,
    preprocess_rgb_image,
)


class DrivingDataset(Dataset):
    def __init__(self, csv_path: Path, image_root: Path, preprocess_cfg: PreprocessConfig) -> None:
        self._samples: List[Tuple[str, float, float]] = []
        self._image_root = image_root
        self._preprocess_cfg = preprocess_cfg

        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._samples.append((row["image_path"], float(row["steering"]), float(row["go"])))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        image_path, steering, go = self._samples[idx]
        path = Path(image_path)
        if not path.is_absolute():
            path = self._image_root / path

        with Image.open(path) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)

        image_tensor = preprocess_rgb_image(rgb, self._preprocess_cfg).squeeze(0)
        steering_tensor = torch.tensor([steering], dtype=torch.float32)
        go_tensor = torch.tensor([go], dtype=torch.float32)
        return image_tensor, steering_tensor, go_tensor


def split_indices(length: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    val_size = max(1, int(length * val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:] if val_size < length else indices
    return train_indices, val_indices


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    lambda_stop: float,
    device: torch.device,
    stop_loss_fn: nn.Module,
    steering_loss_fn: nn.Module,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for images, steering_target, go_target in loader:
            images = images.to(device)
            steering_target = steering_target.to(device)
            go_target = go_target.to(device)

            steering_pred, stop_logit = model(images)
            steering_loss = masked_steering_loss(steering_pred, steering_target, go_target, steering_loss_fn)
            stop_target = 1.0 - go_target
            stop_loss = stop_loss_fn(stop_logit, stop_target)
            loss = steering_loss + lambda_stop * stop_loss

            batch_size = images.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

    return total_loss / max(total_count, 1)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    preprocess_cfg = PreprocessConfig(
        image_width=args.image_width,
        image_height=args.image_height,
        crop_top_pixels=args.crop_top_pixels,
    )

    dataset = DrivingDataset(Path(args.csv), Path(args.image_root), preprocess_cfg)
    if len(dataset) < 4:
        raise ValueError("Need at least 4 samples for train/val split")

    train_indices, val_indices = split_indices(len(dataset), args.val_ratio, args.seed)
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = PilotNetDualHead(image_height=args.image_height, image_width=args.image_width).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stop_loss_fn = nn.BCEWithLogitsLoss()
    steering_loss_fn = nn.SmoothL1Loss(reduction="none")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, steering_target, go_target in train_loader:
            images = images.to(device)
            steering_target = steering_target.to(device)
            go_target = go_target.to(device)

            steering_pred, stop_logit = model(images)
            steering_loss = masked_steering_loss(steering_pred, steering_target, go_target, steering_loss_fn)
            stop_target = 1.0 - go_target
            stop_loss = stop_loss_fn(stop_logit, stop_target)
            loss = steering_loss + args.lambda_stop * stop_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = evaluate(model, val_loader, args.lambda_stop, device, stop_loss_fn, steering_loss_fn)
        print(f"epoch={epoch} val_loss={val_loss:.6f}")

        ckpt = {
            "model_state_dict": model.state_dict(),
            "preprocess": {
                "image_width": args.image_width,
                "image_height": args.image_height,
                "crop_top_pixels": args.crop_top_pixels,
            },
        }

        torch.save(ckpt, output_dir / "model_last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, output_dir / "model_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PilotNet dual-head model for Rosmaster A1 MVP")
    parser.add_argument("--csv", required=True, help="Dataset CSV path")
    parser.add_argument("--image-root", required=True, help="Image root directory")
    parser.add_argument("--output-dir", default="training/checkpoints")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--lambda-stop", type=float, default=1.0)
    parser.add_argument("--image-width", type=int, default=200)
    parser.add_argument("--image-height", type=int, default=66)
    parser.add_argument("--crop-top-pixels", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    train(parser.parse_args())
