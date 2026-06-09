from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch

from .model import PilotNetDualHead, PreprocessConfig, preprocess_rgb_image


class InferenceBackend(ABC):
    @abstractmethod
    def predict(self, image_rgb: np.ndarray) -> Tuple[float, float]:
        raise NotImplementedError


class TorchInferenceBackend(InferenceBackend):
    def __init__(
        self,
        model_path: str,
        preprocess: PreprocessConfig,
        device: str = "cpu",
    ) -> None:
        self._preprocess = preprocess
        self._device = torch.device(device)
        self._model = PilotNetDualHead(
            image_height=preprocess.image_height,
            image_width=preprocess.image_width,
        )

        checkpoint = torch.load(model_path, map_location=self._device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()

    def predict(self, image_rgb: np.ndarray) -> Tuple[float, float]:
        with torch.no_grad():
            tensor = preprocess_rgb_image(image_rgb, self._preprocess).to(self._device)
            steering, stop_logit = self._model(tensor)
            steering_value = float(steering.squeeze(0).item())
            stop_prob = float(torch.sigmoid(stop_logit).squeeze(0).item())
            return steering_value, stop_prob


class OnnxInferenceBackend(InferenceBackend):
    def __init__(self, model_path: str, preprocess: PreprocessConfig) -> None:
        import onnxruntime as ort

        self._preprocess = preprocess
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inputs = self._session.get_inputs()
        self._input_name = inputs[0].name

    def predict(self, image_rgb: np.ndarray) -> Tuple[float, float]:
        tensor = preprocess_rgb_image(image_rgb, self._preprocess).numpy()
        steering, stop_logit = self._session.run(None, {self._input_name: tensor})
        steering_value = float(np.asarray(steering).reshape(-1)[0])
        stop_prob = 1.0 / (1.0 + np.exp(-float(np.asarray(stop_logit).reshape(-1)[0])))
        return steering_value, stop_prob


def create_inference_backend(
    backend: str,
    model_path: str,
    preprocess: PreprocessConfig,
    device: str,
) -> InferenceBackend:
    backend = backend.lower().strip()
    if backend == "torch":
        return TorchInferenceBackend(model_path=model_path, preprocess=preprocess, device=device)
    if backend == "onnx":
        return OnnxInferenceBackend(model_path=model_path, preprocess=preprocess)
    raise ValueError(f"Unsupported model backend '{backend}'")
