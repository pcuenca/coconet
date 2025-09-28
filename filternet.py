"""Inference helpers for the Filternet model."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow<9.1 fallback
    RESAMPLE_BILINEAR = Image.BILINEAR
from fastai.vision.all import Normalize, imagenet_stats, resnet50
from fastai.vision.learner import create_cnn_model
from torch import nn

from magicml.filters.all import (
    EVFilter,
    GradBrightness,
    GradContrast,
    GradHighlights,
    GradShadows,
    TemperatureFilter,
    linear_to_sRGB,
    sRGB_to_linear,
)
from magicml.model import FilterNet

PathLike = Union[str, Path]
DeviceLike = Union[str, torch.device]
Tensor = torch.Tensor

DEFAULT_MAX_KELVIN = 10000
DEFAULT_MIN_KELVIN = 4000
DEFAULT_ARCH = resnet50
DEFAULT_IMAGE_SIZE = 299

FILTER_NAMES = (
    "temperature",
    "ev",
    "brightness",
    "contrast",
    "shadows",
    "highlights",
)


def _resolve_device(device: Optional[DeviceLike] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _default_filters(max_kelvin: int, min_kelvin: int):
    from functools import partial

    ranged_temperature = partial(
        TemperatureFilter, maxKelvin=max_kelvin, minKelvin=min_kelvin
    )
    return (
        ranged_temperature,
        EVFilter,
        GradBrightness,
        GradContrast,
        GradShadows,
        GradHighlights,
    )


def _init_head_small(head: nn.Module) -> None:
    for module in head.modules():
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()


def _build_filternet_model(
    *,
    arch,
    filters,
    norm: Normalize,
    dropout: Optional[float] = None,
    init_small: bool = True,
) -> FilterNet:
    n_out = len(filters)
    base_model = create_cnn_model(arch, n_out, pretrained=False)
    body, head = base_model[0], base_model[1]
    if init_small:
        _init_head_small(head)
    model = nn.Sequential(body, head)
    filter_net = FilterNet(model, filters, norm, dropout)
    return filter_net


def _pil_to_tensor(image: Image.Image, size: Optional[int] = None) -> Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")
    if size is not None:
        image = image.resize((size, size), RESAMPLE_BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor


def _linear_normalize(tensor: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    linear = sRGB_to_linear(tensor)
    return (linear - mean) / std



def apply_filters_to_image(
    image: Image.Image,
    intensities: Sequence[float],
    *,
    filters: Optional[Sequence] = None,
) -> Image.Image:
    """Apply Filternet-style filters to `image` using the provided intensities."""
    if filters is None:
        filters = _default_filters(DEFAULT_MAX_KELVIN, DEFAULT_MIN_KELVIN)

    if len(intensities) != len(filters):
        raise ValueError(
            f"Expected {len(filters)} intensities but received {len(intensities)}."
        )

    tensor = _pil_to_tensor(image)
    tensor = sRGB_to_linear(tensor)
    tensor = tensor.unsqueeze(0)
    for filter_factory, value in zip(filters, intensities):
        module = filter_factory(intensities=[float(value)])
        module = module.to(tensor.device, dtype=tensor.dtype)
        tensor = module(tensor)
    tensor = tensor.squeeze(0).clamp(0.0, 1.0)
    srgb = linear_to_sRGB(tensor).clamp(0.0, 1.0)
    array = (srgb.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array)


@dataclass
class FilternetPrediction:
    image: Image.Image
    filters: Dict[str, float]


class FilternetPredictor:
    """Wrapper around the Filternet model for single-image inference."""

    def __init__(
        self,
        checkpoint_path: PathLike,
        *,
        device: Optional[DeviceLike] = None,
        arch=DEFAULT_ARCH,
        image_size: int = DEFAULT_IMAGE_SIZE,
        max_kelvin: int = DEFAULT_MAX_KELVIN,
        min_kelvin: int = DEFAULT_MIN_KELVIN,
        dropout: Optional[float] = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.image_size = image_size
        self.filters_to_apply = _default_filters(max_kelvin, min_kelvin)
        self.filter_names = FILTER_NAMES

        self._norm = Normalize.from_stats(*imagenet_stats)
        self._norm_mean = self._norm.mean.squeeze(0).to(torch.float32)
        self._norm_std = self._norm.std.squeeze(0).to(torch.float32)

        self.model = _build_filternet_model(
            arch=arch,
            filters=self.filters_to_apply,
            norm=self._norm,
            dropout=dropout,
        ).to(self.device)
        self.model.eval()

        state = torch.load(
            str(checkpoint_path), map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(state["model"])

        self._params: Optional[Tensor] = None
        self._hook = self.model.model.register_forward_hook(self._capture_params)

    def _capture_params(self, module: nn.Module, inputs, output: Tensor) -> None:
        self._params = output.detach().to("cpu", torch.float32)

    def _preprocess(self, image: Image.Image) -> Tensor:
        tensor = _pil_to_tensor(image, self.image_size)
        normalized = _linear_normalize(
            tensor,
            self._norm_mean,
            self._norm_std,
        )
        dtype = next(self.model.parameters()).dtype
        return normalized.unsqueeze(0).to(self.device, dtype)

    def predict(self, image: Image.Image) -> FilternetPrediction:
        if image.width != image.height:
            raise ValueError("Filternet expects square inputs.")

        batch = self._preprocess(image)
        with torch.inference_mode():
            self._params = None
            _ = self.model(batch)
        if self._params is None:
            raise RuntimeError("Failed to capture filter parameters from the model forward pass.")
        params = self._params.squeeze(0)
        intensities = [float(value) for value in params.tolist()]
        predicted = apply_filters_to_image(
            image, intensities, filters=self.filters_to_apply
        )
        filter_map = OrderedDict(zip(self.filter_names, intensities))
        return FilternetPrediction(image=predicted, filters=dict(filter_map))

    def apply_filters(self, image: Image.Image, intensities: Sequence[float]) -> Image.Image:
        return apply_filters_to_image(
            image, intensities, filters=self.filters_to_apply
        )

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __del__(self) -> None:
        self.close()


def load_filternet_predictor(
    checkpoint_path: PathLike,
    *,
    device: Optional[DeviceLike] = None,
    arch=DEFAULT_ARCH,
    image_size: int = DEFAULT_IMAGE_SIZE,
    max_kelvin: int = DEFAULT_MAX_KELVIN,
    min_kelvin: int = DEFAULT_MIN_KELVIN,
    dropout: Optional[float] = None,
) -> FilternetPredictor:
    """Factory helper mirroring the notebook setup."""
    return FilternetPredictor(
        checkpoint_path,
        device=device,
        arch=arch,
        image_size=image_size,
        max_kelvin=max_kelvin,
        min_kelvin=min_kelvin,
        dropout=dropout,
    )


def run_filternet_inference(
    predictor: FilternetPredictor, image: Image.Image
) -> FilternetPrediction:
    """Convenience wrapper to mirror the notebook usage pattern."""
    return predictor.predict(image)
