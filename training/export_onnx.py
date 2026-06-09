from __future__ import annotations

import argparse
from pathlib import Path

import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rosmaster_a1_e2e_vision.model import PilotNetDualHead  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export checkpoint to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-width", type=int, default=200)
    parser.add_argument("--image-height", type=int, default=66)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = PilotNetDualHead(image_height=args.image_height, image_width=args.image_width)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.zeros(1, 3, args.image_height, args.image_width)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["image"],
        output_names=["steering", "stop_logit"],
        opset_version=13,
    )
    print(f"Exported {args.output}")
