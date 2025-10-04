from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDIMScheduler, UNet2DConditionModel
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


# These are simplified for 1 channel
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


def _ensure_sequence(obj, length: Optional[int] = None):
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if length is not None and length != 1:
        raise ValueError("Expected a sequence matching the batch size.")
    return [obj]


@torch.inference_mode()
def run_cocogold_inference(
    pipeline: MarigoldPipeline,
    image: Union[Image.Image, Sequence[Image.Image]],
    prompt: Union[str, Sequence[str]],
    *,
    num_inference_steps: int = 50,
    desaturation_threshold: float = 0.6,
    desaturation_factor: float = 0.7,
    mask_threshold: float = 0.95,
    mask_kernel_size: int = 3,
    ensemble_runs: int = 1,
    ensemble_reduction: str = "median",
    scheduler_name: Optional[str] = None,
    generator: Optional[torch.Generator] = None,
    show_progress: bool = False,
) -> Tuple[Image.Image, Image.Image]:
    images = _ensure_sequence(image)
    prompts = _ensure_sequence(prompt, length=len(images))

    for img in images:
        if img.width != img.height:
            raise ValueError("Expected square images for CocoGold inference.")

    processed = []
    for img in images:
        tensor = _pil_to_tensor(img)
        tensor = desaturate_highlights(
            tensor, threshold=desaturation_threshold, desaturation_factor=desaturation_factor
        )
        processed.append(tensor)

    pipe_input = torch.stack(processed, dim=0).to(device=pipeline.device, dtype=pipeline.unet.dtype)

    tokenized = pipeline.tokenizer(
        prompts,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(pipeline.device)
    text_embeddings = pipeline.text_encoder(tokenized.input_ids)[0]
    text_embeddings = text_embeddings.to(dtype=pipeline.unet.dtype, device=pipeline.device)

    if ensemble_runs < 1:
        raise ValueError("ensemble_runs must be >= 1")
    if ensemble_reduction not in {"mean", "median"}:
        raise ValueError("ensemble_reduction must be 'mean' or 'median'")

    pred_means = []
    predicted_images = []

    base_seed = None
    if generator is not None:
        base_seed = generator.initial_seed()

    original_scheduler = None
    if scheduler_name is not None:
        scheduler_name = scheduler_name.lower()
        if scheduler_name == "trailing_ddim":
            original_scheduler = pipeline.scheduler
            pipeline.scheduler = DDIMScheduler.from_config(
                original_scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
            )
        else:
            raise ValueError(f"Unsupported scheduler_name: {scheduler_name}")

    try:
        for idx in range(ensemble_runs):
            current_generator = generator
            if generator is not None:
                current_generator = torch.Generator(device=pipeline.device)
                current_generator.manual_seed(base_seed + idx)

            pred_mean, predicted = pipeline.single_infer(
                pipe_input,
                text_embeddings,
                num_inference_steps=num_inference_steps,
                generator=current_generator,
                show_pbar=show_progress,
            )

            pred_means.append(pred_mean.cpu())
            predicted_images.append(predicted.cpu())
    finally:
        if original_scheduler is not None:
            pipeline.scheduler = original_scheduler

    pred_means_tensor = torch.stack(pred_means, dim=0)  # [E, B, 1, H, W]
    predicted_tensor = torch.stack(predicted_images, dim=0)  # [E, B, C, H, W]

    if ensemble_runs == 1:
        aggregated_mean = pred_means_tensor[0]
        aggregated_prediction = predicted_tensor[0]
    else:
        if ensemble_reduction == "mean":
            aggregated_mean = pred_means_tensor.mean(dim=0)
            aggregated_prediction = predicted_tensor.mean(dim=0)
        else:
            aggregated_mean = pred_means_tensor.median(dim=0).values
            aggregated_prediction = predicted_tensor.median(dim=0).values

    aggregated_mean = aggregated_mean.squeeze(1)  # [B, H, W]

    batch_size = aggregated_prediction.shape[0]
    predicted_images_pil = []
    mask_images_pil = []

    for idx in range(batch_size):
        predicted_images_pil.append(_tensor_to_pil_image(aggregated_prediction[idx]))
        mask = build_mask(
            aggregated_mean[idx], threshold=mask_threshold, kernel_size=mask_kernel_size
        )
        mask_images_pil.append(_tensor_to_pil_mask(mask))

    if len(predicted_images_pil) == 1:
        return predicted_images_pil[0], mask_images_pil[0]
    return predicted_images_pil, mask_images_pil
