"""Gradio UI for combining Filternet edits with CocoGold masks."""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import gradio as gr
from PIL import Image, ImageDraw

from cocogold_pipeline import load_cocogold_pipeline
from coconet import (
    CombinedFilternetResult,
    apply_filternet_with_cocogold,
    apply_filternet_with_cocogold_bbox,
)
from filternet import FilternetPredictor


FILTERNET_CHECKPOINT = Path(
    "/home/pedro/code/photo-editing/filternet/models/p-tebcwh-batch-norm-moderate-resnet50-noconv-unfrozen.pth"
)
COCOGOLD_CHECKPOINT_DIR = Path(
    "/home/pedro/code/hf/diffusers/marigold-segmentation/Marigold/output/overlapped/25_06_11-13_00_18-train_cocogold/checkpoint"
)
SD_CHECKPOINT = Path(
    "/home/pedro/code/hf/diffusers/marigold-segmentation/checkpoints/stable-diffusion-2"
)

COCOGOLD_CHOICES: Dict[str, Path] = {
    "iter_8000": COCOGOLD_CHECKPOINT_DIR / "iter_008000",
    "iter_018000": COCOGOLD_CHECKPOINT_DIR / "iter_018000",
}

IMAGE_SIZE = 512
MASK_FEATHER_RADIUS = 5.0
BBOX_PADDING = 12


@lru_cache(maxsize=1)
def get_filternet_predictor() -> FilternetPredictor:
    return FilternetPredictor(str(FILTERNET_CHECKPOINT))


@lru_cache(maxsize=None)
def get_cocogold_pipeline(unet_path: str):
    return load_cocogold_pipeline(unet_path, str(SD_CHECKPOINT))


def _prepare_image(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)


def _format_filters(filters: Dict[str, float], names) -> Dict[str, float]:
    ordered = OrderedDict()
    for name in names:
        value = float(filters.get(name, 0.0))
        ordered[name] = round(value, 4)
    return ordered


def _draw_bbox_on_mask(mask: Image.Image, bbox: Optional[Tuple[int, int, int, int]]) -> Image.Image:
    if bbox is None:
        return mask
    color_mask = mask.convert("RGB")
    draw = ImageDraw.Draw(color_mask)
    draw.rectangle(bbox, outline=(0, 255, 0), width=3)
    return color_mask


def _run_pipelines(
    image: Image.Image,
    prompt: str,
    unet_choice: str,
    ensemble_runs: int,
    ensemble_reduction: str,
    num_steps: int,
    scheduler_choice: str,
):
    if image is None or not prompt.strip():
        raise gr.Error("Please provide an image and a prompt before running the demo.")

    processed_image = _prepare_image(image)
    ensemble_runs = int(ensemble_runs)
    num_steps = int(num_steps)
    scheduler_name = None if scheduler_choice == "default" else scheduler_choice

    filternet = get_filternet_predictor()
    unet_path = COCOGOLD_CHOICES.get(unet_choice, COCOGOLD_CHOICES["iter_8000"])
    pipeline = get_cocogold_pipeline(str(unet_path))

    full_result: CombinedFilternetResult = apply_filternet_with_cocogold(
        image=processed_image,
        prompt=prompt,
        filternet=filternet,
        cocogold_pipeline=pipeline,
        mask_feather_radius=MASK_FEATHER_RADIUS,
        ensemble_runs=ensemble_runs,
        ensemble_reduction=ensemble_reduction,
        num_inference_steps=num_steps,
        scheduler_name=scheduler_name,
    )

    bbox_result: CombinedFilternetResult = apply_filternet_with_cocogold_bbox(
        image=processed_image,
        prompt=prompt,
        filternet=filternet,
        cocogold_pipeline=pipeline,
        bbox_padding=BBOX_PADDING,
        mask_feather_radius=MASK_FEATHER_RADIUS,
        mask_image=full_result.mask,
        cocogold_prediction=full_result.cocogold_prediction,
        ensemble_runs=ensemble_runs,
        ensemble_reduction=ensemble_reduction,
        num_inference_steps=num_steps,
        scheduler_name=scheduler_name,
    )

    full_mask = full_result.mask
    bbox_mask = _draw_bbox_on_mask(bbox_result.mask, bbox_result.bbox)

    full_filters = _format_filters(full_result.filternet.filters, filternet.filter_names)
    bbox_filters = _format_filters(bbox_result.filternet.filters, filternet.filter_names)

    return (
        full_result.composited,
        bbox_result.composited,
        full_mask,
        bbox_mask,
        full_filters,
        bbox_filters,
    )


with gr.Blocks(title="CocoNet Demo") as demo:
    gr.Markdown(
        """# CocoNet Demo

Upload an image, provide a short prompt, and compare two Filternet blending strategies driven by CocoGold masks."""
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Input Image", height=350)
        with gr.Column():
            with gr.Row():
                prompt_input = gr.Textbox(label="Prompt", placeholder="Describe what to segment", lines=2)
                unet_choice = gr.Dropdown(
                    choices=list(COCOGOLD_CHOICES.keys()),
                    value="iter_8000",
                    label="UNet checkpoint",
                )
            with gr.Row():
                ensemble_runs = gr.Slider(
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=1,
                    label="Ensemble runs",
                )
                ensemble_reduction = gr.Radio(
                    choices=["median", "mean"],
                    value="median",
                    label="Ensemble reduction",
                )
            with gr.Row():
                num_steps = gr.Slider(
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=50,
                    label="Inference steps",
                )
                scheduler_choice = gr.Dropdown(
                    choices=["default", "trailing_ddim"],
                    value="default",
                    label="Scheduler",
                )
            run_button = gr.Button("Run Pipelines", variant="primary")

    with gr.Row():
        blended_full = gr.Image(label="Full Image Blend", height=300)
        blended_bbox = gr.Image(label="BBox Blend", height=300)

    with gr.Row():
        mask_full = gr.Image(label="Mask", height=250)
        mask_bbox = gr.Image(label="Mask with bbox", height=250)

    with gr.Row():
        filters_full = gr.JSON(label="Filternet Parameters (Full)")
        filters_bbox = gr.JSON(label="Filternet Parameters (BBox)")

    run_button.click(
        _run_pipelines,
        inputs=[image_input, prompt_input, unet_choice, ensemble_runs, ensemble_reduction, num_steps, scheduler_choice],
        outputs=[blended_full, blended_bbox, mask_full, mask_bbox, filters_full, filters_bbox],
    )


if __name__ == "__main__":
    demo.launch()
