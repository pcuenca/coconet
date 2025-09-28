"""High-level helpers that blend Filternet edits using CocoGold segmentation masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from PIL import Image, ImageFilter

from cocogold import run_cocogold_inference
from filternet import FilternetPrediction, FilternetPredictor


@dataclass
class CombinedFilternetResult:
    composited: Image.Image
    filternet: FilternetPrediction
    mask: Image.Image
    cocogold_prediction: Image.Image


def _combine_with_mask(
    original: Image.Image,
    filtered: Image.Image,
    mask: Image.Image,
    *,
    feather_radius: float = 0.0,
) -> Image.Image:
    if feather_radius > 0.0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    base = np.asarray(original.convert("RGB"), dtype=np.float32)
    overlay = np.asarray(filtered.convert("RGB"), dtype=np.float32)
    weights = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    weights = np.expand_dims(weights, axis=-1)
    blended = overlay * weights + base * (1.0 - weights)
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(blended)


def apply_filternet_with_cocogold(
    *,
    image: Image.Image,
    prompt: str,
    filternet: FilternetPredictor,
    cocogold_pipeline,
    num_inference_steps: int = 50,
    desaturation_threshold: float = 0.6,
    desaturation_factor: float = 0.7,
    mask_threshold: float = 0.95,
    mask_kernel_size: int = 3,
    mask_feather_radius: float = 0.0,
    generator=None,
    show_progress: bool = False,
) -> CombinedFilternetResult:
    filternet_result = filternet.predict(image)

    predicted_image, mask_image = run_cocogold_inference(
        cocogold_pipeline,
        image,
        prompt,
        num_inference_steps=num_inference_steps,
        desaturation_threshold=desaturation_threshold,
        desaturation_factor=desaturation_factor,
        mask_threshold=mask_threshold,
        mask_kernel_size=mask_kernel_size,
        generator=generator,
        show_progress=show_progress,
    )

    composited = _combine_with_mask(
        image,
        filternet_result.image,
        mask_image,
        feather_radius=mask_feather_radius,
    )

    return CombinedFilternetResult(
        composited=composited,
        filternet=filternet_result,
        mask=mask_image,
        cocogold_prediction=predicted_image,
    )
