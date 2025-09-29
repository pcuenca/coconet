"""Evaluation script for CocoGold segmentation predictions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from marigold.marigold_pipeline import MarigoldPipeline

from cocogold import load_cocogold_pipeline, run_cocogold_inference
from cocogold.dataset import CocoGoldIterableDataset


def _to_bool_mask(mask_image: Image.Image) -> np.ndarray:
    grayscale = mask_image.convert("L")
    array = np.asarray(grayscale, dtype=np.uint8)
    return array > 127


def _update_confusion(stats: Dict[str, int], pred: np.ndarray, target: np.ndarray) -> None:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, np.logical_not(target)).sum()
    fn = np.logical_and(np.logical_not(pred), target).sum()
    tn = pred.size - tp - fp - fn
    stats["tp"] += int(tp)
    stats["fp"] += int(fp)
    stats["fn"] += int(fn)
    stats["tn"] += int(tn)


def _metrics_from_confusion(stats: Dict[str, int]) -> Dict[str, float]:
    tp = stats["tp"]
    fp = stats["fp"]
    fn = stats["fn"]
    tn = stats["tn"]
    total = tp + fp + fn + tn

    metrics: Dict[str, float] = {}
    if total == 0:
        return {"iou": 0.0, "pixel_acc": 0.0, "precision": 0.0, "recall": 0.0, "dice": 0.0}

    precision_den = tp + fp
    recall_den = tp + fn
    iou_den = tp + fp + fn
    dice_den = 2 * tp + fp + fn

    precision = tp / precision_den if precision_den > 0 else 0.0
    recall = tp / recall_den if recall_den > 0 else 0.0
    iou = tp / iou_den if iou_den > 0 else 0.0
    dice = (2 * tp) / dice_den if dice_den > 0 else 0.0
    pixel_acc = (tp + tn) / total if total > 0 else 0.0

    metrics.update(
        {
            "iou": iou,
            "pixel_acc": pixel_acc,
            "precision": precision,
            "recall": recall,
            "dice": dice,
        }
    )
    return metrics


def _build_bucket_sets(
    trained_categories: Optional[Sequence[str]],
    all_categories: Sequence[str],
) -> Dict[str, set]:
    if trained_categories is None:
        return {}
    trained_set = set(trained_categories)
    other_set = set(all_categories) - trained_set
    return {
        "trained": trained_set,
        "other": other_set,
    }


def evaluate_cocogold(
    dataset_root: Path,
    *,
    split: str = "val",
    categories: Optional[Sequence[str]] = None,
    trained_categories: Optional[Sequence[str]] = None,
    batch_size: int = 1,
    max_items: Optional[int] = None,
    ensemble_runs: int = 1,
    ensemble_reduction: str = "median",
    unet_checkpoint: Path,
    sd_checkpoint: Path,
    device: Optional[str] = None,
    torch_dtype: Optional[str] = None,
    seed: Optional[int] = None,
    progress: bool = True,
    num_inference_steps: int = 50,
    mask_threshold: float = 0.95,
    mask_kernel_size: int = 3,
    desaturation_threshold: float = 0.6,
    desaturation_factor: float = 0.7,
    prompt_template: str = "{category}",
    pipeline: Optional[MarigoldPipeline] = None,
    scheduler_name: Optional[str] = None,
) -> Dict[str, object]:
    dataset = CocoGoldIterableDataset(
        dataset_root,
        split=split,
        size=512,
        max_items=max_items,
        valid_cat_names=categories,
        return_type="pil",
        seed=seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )

    local_pipeline = pipeline is None
    if pipeline is None:
        resolved_dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        pipeline = load_cocogold_pipeline(
            unet_checkpoint,
            sd_checkpoint,
            device=device,
            torch_dtype=resolved_dtype,
        )

    pipeline_dtype = str(getattr(pipeline.unet, "dtype", "unknown"))

    confusion_overall: Dict[str, int] = defaultdict(int)
    confusion_by_category: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    available_categories = [cat["name"] for cat in dataset.dataset.cats]
    bucket_sets = _build_bucket_sets(trained_categories, available_categories)
    confusion_by_bucket: Dict[str, Dict[str, int]] = {name: defaultdict(int) for name in bucket_sets}
    counts_by_category: Dict[str, int] = defaultdict(int)
    counts_by_bucket: Dict[str, int] = defaultdict(int)
    total_samples = 0

    iterator = dataloader
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=len(dataloader))
        except Exception:
            pass

    generator_seed = seed if seed is not None else None
    sample_index = 0

    for batch in iterator:
        for sample in batch:
            image: Image.Image = sample["image"]
            target_mask: Image.Image = sample["mask"]
            category = sample["class"]

            prompt = prompt_template.format(category=category)

            current_generator = None
            if generator_seed is not None:
                current_generator = torch.Generator(device=pipeline.device)
                current_generator.manual_seed(generator_seed + sample_index)

            predicted_image, predicted_mask_image = run_cocogold_inference(
                pipeline,
                image,
                prompt,
                num_inference_steps=num_inference_steps,
                desaturation_threshold=desaturation_threshold,
                desaturation_factor=desaturation_factor,
                mask_threshold=mask_threshold,
                mask_kernel_size=mask_kernel_size,
                ensemble_runs=ensemble_runs,
                ensemble_reduction=ensemble_reduction,
                scheduler_name=scheduler_name,
                generator=current_generator,
                show_progress=False,
            )

            pred_mask = _to_bool_mask(predicted_mask_image)
            target = _to_bool_mask(target_mask)

            _update_confusion(confusion_overall, pred_mask, target)
            _update_confusion(confusion_by_category[category], pred_mask, target)
            counts_by_category[category] += 1
            total_samples += 1

            for bucket_name, category_set in bucket_sets.items():
                if category in category_set:
                    _update_confusion(confusion_by_bucket[bucket_name], pred_mask, target)
                    counts_by_bucket[bucket_name] += 1

            sample_index += 1

    overall_metrics = _metrics_from_confusion(confusion_overall)
    overall_metrics["count"] = total_samples

    by_category_metrics = {
        cat: {**_metrics_from_confusion(stats), "count": counts_by_category[cat]}
        for cat, stats in confusion_by_category.items()
    }

    by_bucket_metrics = {
        name: {**_metrics_from_confusion(stats), "count": counts_by_bucket.get(name, 0)}
        for name, stats in confusion_by_bucket.items()
    }

    results: Dict[str, Dict[str, float]] = {
        "overall": overall_metrics,
        "by_category": by_category_metrics,
        "by_bucket": by_bucket_metrics,
        "config": {
            "split": split,
            "categories": list(categories) if categories is not None else None,
            "trained_categories": list(trained_categories) if trained_categories is not None else None,
            "batch_size": batch_size,
            "max_items": max_items,
            "ensemble_runs": ensemble_runs,
            "ensemble_reduction": ensemble_reduction,
            "num_inference_steps": num_inference_steps,
            "mask_threshold": mask_threshold,
            "mask_kernel_size": mask_kernel_size,
            "desaturation_threshold": desaturation_threshold,
            "desaturation_factor": desaturation_factor,
            "prompt_template": prompt_template,
            "scheduler": scheduler_name,
            "unet_checkpoint": str(unet_checkpoint),
            "sd_checkpoint": str(sd_checkpoint),
            "device": device,
            "torch_dtype": torch_dtype,
            "pipeline_dtype": pipeline_dtype,
            "seed": seed,
        },
    }

    if local_pipeline:
        del pipeline

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CocoGold segmentation model.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("unet_checkpoint", type=Path)
    parser.add_argument("sd_checkpoint", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--trained-categories", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--ensemble-runs", type=int, default=1)
    parser.add_argument("--ensemble-reduction", choices=["mean", "median"], default="median")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--mask-threshold", type=float, default=0.95)
    parser.add_argument("--mask-kernel-size", type=int, default=3)
    parser.add_argument("--desaturation-threshold", type=float, default=0.6)
    parser.add_argument("--desaturation-factor", type=float, default=0.7)
    parser.add_argument("--prompt-template", default="{category}")
    parser.add_argument("--scheduler", default=None, help="Optional scheduler override (e.g. trailing_ddim)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to store JSON results.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = evaluate_cocogold(
        args.dataset_root,
        split=args.split,
        categories=args.categories,
        trained_categories=args.trained_categories,
        batch_size=args.batch_size,
        max_items=args.max_items,
        ensemble_runs=args.ensemble_runs,
        ensemble_reduction=args.ensemble_reduction,
        num_inference_steps=args.num_inference_steps,
        mask_threshold=args.mask_threshold,
        mask_kernel_size=args.mask_kernel_size,
        desaturation_threshold=args.desaturation_threshold,
        desaturation_factor=args.desaturation_factor,
        prompt_template=args.prompt_template,
        scheduler_name=args.scheduler,
        unet_checkpoint=args.unet_checkpoint,
        sd_checkpoint=args.sd_checkpoint,
        device=args.device,
        torch_dtype=args.torch_dtype,
        seed=args.seed,
        progress=not args.no_progress,
    )

    print(json.dumps(results, indent=2))
    if args.output:
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
