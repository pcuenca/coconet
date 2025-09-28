"""Utilities for running CocoGold segmentation inference outside the notebook workflow."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import UNet2DConditionModel
from marigold.marigold_pipeline import MarigoldPipeline

Tensor = torch.Tensor
PathLike = Union[str, Path]
DeviceLike = Union[str, torch.device]


def _resolve_device(device: Optional[DeviceLike] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_cocogold_pipeline(
    unet_checkpoint: PathLike,
    sd_checkpoint: PathLike,
    *,
    device: Optional[DeviceLike] = None,
    torch_dtype: Optional[torch.dtype] = None,
) -> MarigoldPipeline:
    """Instantiate a Marigold pipeline using explicit checkpoint paths."""
    device = _resolve_device(device)
    if torch_dtype is None:
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    unet = UNet2DConditionModel.from_pretrained(
        str(unet_checkpoint), subfolder="unet", torch_dtype=torch_dtype
    )
    pipe = MarigoldPipeline.from_pretrained(
        str(sd_checkpoint),
        unet=unet,
        torch_dtype=torch_dtype,
        scale_invariant=True,
        shift_invariant=True,
    )
    pipe = pipe.to(device)
    return pipe


@torch.inference_mode()
def encode_prompt(pipe: MarigoldPipeline, prompt: str) -> Tensor:
    """Encode a text prompt into CLIP embeddings used by Marigold."""
    tokenized = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(pipe.device)
    embeddings = pipe.text_encoder(tokenized.input_ids)[0]
    return embeddings.to(dtype=pipe.unet.dtype, device=pipe.device)


def _pil_to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def _tensor_to_pil_image(image: Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array)


def _tensor_to_pil_mask(mask: Tensor) -> Image.Image:
    mask = mask.detach().cpu().clamp(0.0, 1.0)
    array = (mask.numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def desaturate_highlights(
    image: Tensor, *, threshold: float = 0.6, desaturation_factor: float = 0.7
) -> Tensor:
    highlights = torch.where(
        image > threshold,
        torch.tensor(1.0, dtype=image.dtype, device=image.device),
        torch.tensor(0.0, dtype=image.dtype, device=image.device),
    )
    highlight_mask = (highlights.sum(dim=0) == image.shape[0]).to(image.dtype)
    scaling = torch.where(
        highlight_mask == 1,
        torch.tensor(desaturation_factor, dtype=image.dtype, device=image.device),
        torch.tensor(1.0, dtype=image.dtype, device=image.device),
    )
    scaling = scaling.unsqueeze(0)
    return image * scaling


def _binary_erosion(mask: Tensor, kernel_size: int) -> Tensor:
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32, device=mask.device)
    padding = kernel_size // 2
    mask_ = mask.unsqueeze(0).unsqueeze(0).float()
    convolved = F.conv2d(mask_, kernel, padding=padding)
    eroded = (convolved >= kernel.numel()).float()
    return eroded.squeeze()


def _binary_dilation(mask: Tensor, kernel_size: int) -> Tensor:
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32, device=mask.device)
    padding = kernel_size // 2
    mask_ = mask.unsqueeze(0).unsqueeze(0).float()
    convolved = F.conv2d(mask_, kernel, padding=padding)
    dilated = (convolved > 0).float()
    return dilated.squeeze()


def morphological_opening(mask: Tensor, kernel_size: int = 3) -> Tensor:
    """Perform morphological opening on a binary mask."""
    eroded = _binary_erosion(mask, kernel_size)
    opened = _binary_dilation(eroded, kernel_size)
    return opened


def build_mask(
    prediction_mean: Tensor,
    *,
    threshold: float = 0.95,
    kernel_size: int = 3,
) -> Tensor:
    """Extract a clean binary mask from the diffusion output."""
    mask = (prediction_mean.squeeze().float() > threshold).float()
    return morphological_opening(mask, kernel_size=kernel_size)


def highlight_prediction_with_mask(prediction: Tensor, mask: Tensor) -> Tensor:
    highlighted = prediction.clone()
    mask_expanded = mask.unsqueeze(0).to(highlighted.device)
    highlighted = torch.where(mask_expanded > 0.5, torch.tensor(1.0, device=highlighted.device, dtype=highlighted.dtype), highlighted)
    return highlighted


@torch.inference_mode()
def run_cocogold_inference(
    pipeline: MarigoldPipeline,
    image: Image.Image,
    prompt: str,
    *,
    num_inference_steps: int = 50,
    desaturation_threshold: float = 0.6,
    desaturation_factor: float = 0.7,
    mask_threshold: float = 0.95,
    mask_kernel_size: int = 3,
    generator: Optional[torch.Generator] = None,
    show_progress: bool = False,
) -> Tuple[Image.Image, Image.Image]:
    """Run inference on a single square image and return a highlighted prediction and mask."""
    if image.width != image.height:
        raise ValueError("Expected a square image for CocoGold inference.")

    tensor = _pil_to_tensor(image)
    tensor = desaturate_highlights(
        tensor, threshold=desaturation_threshold, desaturation_factor=desaturation_factor
    )
    tensor = tensor.unsqueeze(0).to(device=pipeline.device, dtype=pipeline.unet.dtype)

    text_embeddings = encode_prompt(pipeline, prompt)

    pred_mean, predicted = pipeline.single_infer(
        tensor,
        text_embeddings,
        num_inference_steps=num_inference_steps,
        generator=generator,
        show_pbar=show_progress,
    )

    pred_mean = pred_mean.squeeze().cpu()
    mask = build_mask(
        pred_mean, threshold=mask_threshold, kernel_size=mask_kernel_size
    )

    prediction = predicted.squeeze().cpu()
    highlighted = highlight_prediction_with_mask(prediction, mask)

    predicted_image = _tensor_to_pil_image(highlighted)
    mask_image = _tensor_to_pil_mask(mask)
    return predicted_image, mask_image
