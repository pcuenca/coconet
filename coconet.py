"""Blend Filternet edits using CocoGold segmentation masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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
    bbox: Optional[Tuple[int, int, int, int]] = None


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


def _mask_to_bbox(mask: Image.Image, *, padding: int = 0) -> Optional[Tuple[int, int, int, int]]:
    array = np.array(mask.convert("L"))
    ys, xs = np.nonzero(array)
    if len(xs) == 0 or len(ys) == 0:
        return None
    left, right = xs.min(), xs.max()
    top, bottom = ys.min(), ys.max()
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(mask.width, right + padding + 1)
    bottom = min(mask.height, bottom + padding + 1)
    return left, top, right, bottom


def _square_resize(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.BILINEAR)


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
    ensemble_runs: int = 1,
    ensemble_reduction: str = "median",
    generator=None,
    show_progress: bool = False,
    mask_image: Optional[Image.Image] = None,
    cocogold_prediction: Optional[Image.Image] = None,
) -> CombinedFilternetResult:
    filternet_result = filternet.predict(image)
    if mask_image is None or cocogold_prediction is None:
        cocogold_prediction, mask_image = run_cocogold_inference(
            cocogold_pipeline,
            image,
            prompt,
            num_inference_steps=num_inference_steps,
            desaturation_threshold=desaturation_threshold,
            desaturation_factor=desaturation_factor,
            mask_threshold=mask_threshold,
            mask_kernel_size=mask_kernel_size,
            ensemble_runs=ensemble_runs,
            ensemble_reduction=ensemble_reduction,
            generator=generator,
            show_progress=show_progress,
        )

    if mask_image.size != image.size:
        raise ValueError("Provided mask must match the input image dimensions.")

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
        cocogold_prediction=cocogold_prediction,
        bbox=None,
    )


def apply_filternet_with_cocogold_bbox(
    *,
    image: Image.Image,
    prompt: str,
    filternet: FilternetPredictor,
    cocogold_pipeline,
    bbox_padding: int = 0,
    num_inference_steps: int = 50,
    desaturation_threshold: float = 0.6,
    desaturation_factor: float = 0.7,
    mask_threshold: float = 0.95,
    mask_kernel_size: int = 3,
    mask_feather_radius: float = 0.0,
    ensemble_runs: int = 1,
    ensemble_reduction: str = "median",
    generator=None,
    show_progress: bool = False,
    mask_image: Optional[Image.Image] = None,
    cocogold_prediction: Optional[Image.Image] = None,
) -> CombinedFilternetResult:
    if mask_image is None or cocogold_prediction is None:
        cocogold_prediction, mask_image = run_cocogold_inference(
            cocogold_pipeline,
            image,
            prompt,
            num_inference_steps=num_inference_steps,
            desaturation_threshold=desaturation_threshold,
            desaturation_factor=desaturation_factor,
            mask_threshold=mask_threshold,
            mask_kernel_size=mask_kernel_size,
            ensemble_runs=ensemble_runs,
            ensemble_reduction=ensemble_reduction,
            generator=generator,
            show_progress=show_progress,
        )

    if mask_image.size != image.size:
        raise ValueError("Provided mask must match the input image dimensions.")

    bbox = _mask_to_bbox(mask_image, padding=bbox_padding)
    if bbox is None:
        return CombinedFilternetResult(
            composited=image.copy(),
            filternet=filternet.predict(image),
            mask=mask_image,
            cocogold_prediction=cocogold_prediction,
            bbox=None,
        )

    left, top, right, bottom = bbox
    crop = image.crop((left, top, right, bottom))
    resized_crop = _square_resize(crop, filternet.image_size)
    crop_prediction = filternet.predict(resized_crop)
    intensities = [float(crop_prediction.filters[name]) for name in filternet.filter_names]
    filtered_crop = filternet.apply_filters(crop, intensities)

    filtered_full = image.copy()
    filtered_full.paste(filtered_crop, (left, top))

    composited = _combine_with_mask(
        image,
        filtered_full,
        mask_image,
        feather_radius=mask_feather_radius,
    )

    return CombinedFilternetResult(
        composited=composited,
        filternet=crop_prediction,
        mask=mask_image,
        cocogold_prediction=cocogold_prediction,
        bbox=bbox,
    )
