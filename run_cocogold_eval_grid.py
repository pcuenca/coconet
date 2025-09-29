"""Utility script to run CocoGold evaluation across multiple configurations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence

import torch

from cocogold_pipeline import load_cocogold_pipeline
from eval_cocogold import evaluate_cocogold


TRAINED_CATEGORIES: Sequence[str] = (
    "car",
    "dining table",
    "chair",
    "train",
    "airplane",
    "giraffe",
    "clock",
    "toilet",
    "bed",
    "bird",
    "truck",
    "cat",
    "horse",
    "dog",
)


def _write_header_if_needed(csv_path: Path, fieldnames: Iterable[str]) -> None:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def _extract_metric(metrics: dict, key: str) -> float:
    if not metrics:
        return 0.0
    return float(metrics.get(key, 0.0))


def run_grid(
    *,
    dataset_root: Path,
    sd_checkpoint: Path,
    checkpoint_dir: Path,
    output_csv: Path,
    models: Sequence[str],
    ensemble_runs_list: Sequence[int],
    ensemble_reduction: str = "median",
    batch_size: int = 1,
    max_items: int | None = None,
    num_inference_steps: int = 50,
    mask_threshold: float = 0.95,
    mask_kernel_size: int = 3,
    desaturation_threshold: float = 0.6,
    desaturation_factor: float = 0.7,
    prompt_template: str = "{category}",
    device: str | None = None,
    torch_dtype: str | None = None,
    seed: int | None = None,
    scheduler: str | None = None,
    extra_trailing_model: str | None = None,
    extra_trailing_steps: Sequence[int] | None = None,
    extra_trailing_ensemble_runs: int = 1,
) -> None:
    if extra_trailing_model and not extra_trailing_steps:
        extra_trailing_steps = (1, 4, 20)

    fieldnames = [
        "model",
        "ensemble_runs",
        "ensemble_reduction",
        "scheduler",
        "num_steps",
        "overall_iou",
        "overall_pixel_acc",
        "overall_dice",
        "overall_precision",
        "overall_recall",
        "overall_count",
        "trained_iou",
        "trained_pixel_acc",
        "trained_dice",
        "trained_count",
        "other_iou",
        "other_pixel_acc",
        "other_dice",
        "other_count",
    ]

    _write_header_if_needed(output_csv, fieldnames)

    with output_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        for model_name in models:
            unet_checkpoint = checkpoint_dir / model_name
            resolved_dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
            pipeline = load_cocogold_pipeline(
                unet_checkpoint,
                sd_checkpoint,
                device=device,
                torch_dtype=resolved_dtype,
            )

            for ensemble_runs in ensemble_runs_list:
                results = evaluate_cocogold(
                    dataset_root,
                    split="val",
                    categories=None,
                    trained_categories=TRAINED_CATEGORIES,
                    batch_size=batch_size,
                    max_items=max_items,
                    ensemble_runs=ensemble_runs,
                    ensemble_reduction=ensemble_reduction,
                    unet_checkpoint=unet_checkpoint,
                    sd_checkpoint=sd_checkpoint,
                    device=device,
                    torch_dtype=torch_dtype,
                    seed=seed,
                    progress=True,
                    num_inference_steps=num_inference_steps,
                    mask_threshold=mask_threshold,
                    mask_kernel_size=mask_kernel_size,
                    desaturation_threshold=desaturation_threshold,
                    desaturation_factor=desaturation_factor,
                    prompt_template=prompt_template,
                    pipeline=pipeline,
                    scheduler_name=scheduler,
                )

                overall = results.get("overall", {})
                buckets = results.get("by_bucket", {})
                trained_metrics = buckets.get("trained", {})
                other_metrics = buckets.get("other", {})

                row = {
                    "model": model_name,
                    "ensemble_runs": ensemble_runs,
                    "ensemble_reduction": ensemble_reduction,
                    "scheduler": scheduler or "default",
                    "num_steps": num_inference_steps,
                    "overall_iou": _extract_metric(overall, "iou"),
                    "overall_pixel_acc": _extract_metric(overall, "pixel_acc"),
                    "overall_dice": _extract_metric(overall, "dice"),
                    "overall_precision": _extract_metric(overall, "precision"),
                    "overall_recall": _extract_metric(overall, "recall"),
                    "overall_count": int(overall.get("count", 0)),
                    "trained_iou": _extract_metric(trained_metrics, "iou"),
                    "trained_pixel_acc": _extract_metric(trained_metrics, "pixel_acc"),
                    "trained_dice": _extract_metric(trained_metrics, "dice"),
                    "trained_count": int(trained_metrics.get("count", 0)),
                    "other_iou": _extract_metric(other_metrics, "iou"),
                    "other_pixel_acc": _extract_metric(other_metrics, "pixel_acc"),
                    "other_dice": _extract_metric(other_metrics, "dice"),
                    "other_count": int(other_metrics.get("count", 0)),
                }

                writer.writerow(row)
                f.flush()

            del pipeline

        if extra_trailing_model and extra_trailing_steps:
            model_name = extra_trailing_model
            unet_checkpoint = checkpoint_dir / model_name
            resolved_dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
            pipeline = load_cocogold_pipeline(
                unet_checkpoint,
                sd_checkpoint,
                device=device,
                torch_dtype=resolved_dtype,
            )

            for steps in extra_trailing_steps:
                results = evaluate_cocogold(
                    dataset_root,
                    split="val",
                    categories=None,
                    trained_categories=TRAINED_CATEGORIES,
                    batch_size=batch_size,
                    max_items=max_items,
                    ensemble_runs=extra_trailing_ensemble_runs,
                    ensemble_reduction=ensemble_reduction,
                    unet_checkpoint=unet_checkpoint,
                    sd_checkpoint=sd_checkpoint,
                    device=device,
                    torch_dtype=torch_dtype,
                    seed=seed,
                    progress=False,
                    num_inference_steps=steps,
                    mask_threshold=mask_threshold,
                    mask_kernel_size=mask_kernel_size,
                    desaturation_threshold=desaturation_threshold,
                    desaturation_factor=desaturation_factor,
                    prompt_template=prompt_template,
                    pipeline=pipeline,
                    scheduler_name="trailing_ddim",
                )

                overall = results.get("overall", {})
                buckets = results.get("by_bucket", {})
                trained_metrics = buckets.get("trained", {})
                other_metrics = buckets.get("other", {})

                row = {
                    "model": f"{model_name}-trailing",
                    "ensemble_runs": extra_trailing_ensemble_runs,
                    "ensemble_reduction": ensemble_reduction,
                    "scheduler": "trailing_ddim",
                    "num_steps": steps,
                    "overall_iou": _extract_metric(overall, "iou"),
                    "overall_pixel_acc": _extract_metric(overall, "pixel_acc"),
                    "overall_dice": _extract_metric(overall, "dice"),
                    "overall_precision": _extract_metric(overall, "precision"),
                    "overall_recall": _extract_metric(overall, "recall"),
                    "overall_count": int(overall.get("count", 0)),
                    "trained_iou": _extract_metric(trained_metrics, "iou"),
                    "trained_pixel_acc": _extract_metric(trained_metrics, "pixel_acc"),
                    "trained_dice": _extract_metric(trained_metrics, "dice"),
                    "trained_count": int(trained_metrics.get("count", 0)),
                    "other_iou": _extract_metric(other_metrics, "iou"),
                    "other_pixel_acc": _extract_metric(other_metrics, "pixel_acc"),
                    "other_dice": _extract_metric(other_metrics, "dice"),
                    "other_count": int(other_metrics.get("count", 0)),
                }

                writer.writerow(row)
                f.flush()

            del pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a grid of CocoGold evaluations and append to CSV.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("sd_checkpoint", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--models",
        nargs="*",
        default=("iter_008000", "iter_018000"),
        help="UNet checkpoint folder names to evaluate.",
    )
    parser.add_argument(
        "--ensemble-runs",
        nargs="*",
        type=int,
        default=(1, 2, 4, 8),
        help="Number of ensemble runs to evaluate.",
    )
    parser.add_argument("--ensemble-reduction", choices=["mean", "median"], default="median")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--mask-threshold", type=float, default=0.95)
    parser.add_argument("--mask-kernel-size", type=int, default=3)
    parser.add_argument("--desaturation-threshold", type=float, default=0.6)
    parser.add_argument("--desaturation-factor", type=float, default=0.7)
    parser.add_argument("--prompt-template", default="{category}")
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--extra-trailing-model", default=None)
    parser.add_argument("--extra-trailing-steps", nargs="*", type=int, default=None)
    parser.add_argument("--extra-trailing-ensemble-runs", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_grid(
        dataset_root=args.dataset_root,
        sd_checkpoint=args.sd_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        output_csv=args.output_csv,
        models=args.models,
        ensemble_runs_list=args.ensemble_runs,
        ensemble_reduction=args.ensemble_reduction,
        batch_size=args.batch_size,
        max_items=args.max_items,
        num_inference_steps=args.num_inference_steps,
        mask_threshold=args.mask_threshold,
        mask_kernel_size=args.mask_kernel_size,
        desaturation_threshold=args.desaturation_threshold,
        desaturation_factor=args.desaturation_factor,
        prompt_template=args.prompt_template,
        device=args.device,
        torch_dtype=args.torch_dtype,
        seed=args.seed,
        scheduler=args.scheduler,
        extra_trailing_model=args.extra_trailing_model,
        extra_trailing_steps=args.extra_trailing_steps,
        extra_trailing_ensemble_runs=args.extra_trailing_ensemble_runs,
    )


if __name__ == "__main__":
    main()
