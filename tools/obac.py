"""Compute the revised Object-Background Activation Contrast diagnostic.

This public implementation excludes every annotated/ignored box and letterbox
padding from the background pool. Its values are therefore not numerically
interchangeable with legacy OBAC tables produced by earlier private scripts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO


def parse_visdrone_annotation(path: Path) -> tuple[list[list[float]], list[list[float]]]:
    objects: list[list[float]] = []
    excluded: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split(",")
        if len(values) < 8:
            continue
        x, y, width, height = map(float, values[:4])
        score, category = int(float(values[4])), int(float(values[5]))
        if min(width, height) <= 0:
            continue
        box = [x, y, x + width, y + height]
        excluded.append(box)
        if score > 0 and 1 <= category <= 10:
            objects.append(box)
    return objects, excluded


def project_box(box, image_width, image_height, feature_width, feature_height):
    ratio = min(feature_width / image_width, feature_height / image_height)
    pad_x = (feature_width - image_width * ratio) / 2
    pad_y = (feature_height - image_height * ratio) / 2
    x1, y1, x2, y2 = box
    fx1 = max(0, min(feature_width - 1, math.floor(x1 * ratio + pad_x)))
    fy1 = max(0, min(feature_height - 1, math.floor(y1 * ratio + pad_y)))
    fx2 = max(fx1 + 1, min(feature_width, math.ceil(x2 * ratio + pad_x)))
    fy2 = max(fy1 + 1, min(feature_height, math.ceil(y2 * ratio + pad_y)))
    return fx1, fy1, fx2, fy2


def mask_box(mask: np.ndarray, box) -> None:
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True


def contrast(activation, objects, excluded, width, height, seed, eps=1e-6):
    feature_height, feature_width = activation.shape
    object_mask = np.zeros((feature_height, feature_width), dtype=bool)
    blocked_mask = np.zeros_like(object_mask)
    for box in objects:
        mask_box(object_mask, project_box(box, width, height, feature_width, feature_height))
    for box in excluded:
        mask_box(blocked_mask, project_box(box, width, height, feature_width, feature_height))

    ratio = min(feature_width / width, feature_height / height)
    pad_x = (feature_width - width * ratio) / 2
    pad_y = (feature_height - height * ratio) / 2
    valid = np.zeros_like(object_mask)
    valid[
        math.ceil(pad_y) : math.floor(pad_y + height * ratio),
        math.ceil(pad_x) : math.floor(pad_x + width * ratio),
    ] = True
    object_mask &= valid
    foreground = np.flatnonzero(object_mask)
    background = np.flatnonzero(valid & ~blocked_mask & ~object_mask)
    if not len(foreground) or not len(background):
        return None

    rng = np.random.default_rng(seed)
    count = min(len(foreground), len(background))
    foreground = rng.choice(foreground, count, replace=False)
    background = rng.choice(background, count, replace=False)
    object_mean = float(activation.flat[foreground].mean())
    background_mean = float(activation.flat[background].mean())
    return object_mean, background_mean, object_mean / (background_mean + eps), count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", required=True, help="VisDrone validation image directory")
    parser.add_argument("--annotations", required=True, help="Raw VisDrone annotation directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--max-images", type=int, default=0, help="0 uses every image")
    parser.add_argument("--sampling-seeds", type=int, nargs="+", default=(0, 1, 2))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    images = Path(args.images)
    annotations = Path(args.annotations)
    model = YOLO(args.weights)
    prediction_layers = list(model.model.model[-1].f)
    features: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index in prediction_layers:
        def hook(_, __, result, layer_index=layer_index):
            if not torch.is_tensor(result) or result.ndim != 4:
                raise ValueError("Expected a BCHW prediction-input feature")
            features[layer_index] = result.detach().float().cpu()

        handles.append(model.model.model[layer_index].register_forward_hook(hook))

    image_paths = sorted(images.glob("*.jpg"))
    order = np.random.default_rng(20261102).permutation(len(image_paths))
    if args.max_images > 0:
        order = order[: args.max_images]
    rows = []
    try:
        for index in order:
            image_path = image_paths[int(index)]
            annotation_path = annotations / f"{image_path.stem}.txt"
            if not annotation_path.is_file():
                raise FileNotFoundError(annotation_path)
            with Image.open(image_path) as image:
                width, height = image.size
            objects, excluded = parse_visdrone_annotation(annotation_path)
            resize_ratio = min(args.imgsz / width, args.imgsz / height)
            small_objects = [
                box for box in objects if max(box[2] - box[0], box[3] - box[1]) * resize_ratio < 32
            ]
            if not small_objects:
                continue
            features.clear()
            model.predict(str(image_path), imgsz=args.imgsz, rect=False, device=args.device, half=False, verbose=False)
            for layer_index, tensor in features.items():
                activation = tensor.abs().mean(1)[0].numpy()
                feature_height, feature_width = activation.shape
                for sampling_seed in args.sampling_seeds:
                    stable_seed = int(
                        hashlib.sha256(f"{image_path.stem}:{sampling_seed}".encode()).hexdigest()[:8], 16
                    )
                    value = contrast(
                        activation, small_objects, excluded, width, height, stable_seed
                    )
                    if value is None:
                        continue
                    object_mean, background_mean, obac, cells = value
                    rows.append(
                        {
                            "image": image_path.stem,
                            "layer": layer_index,
                            "stride": args.imgsz / feature_width,
                            "feature_h": feature_height,
                            "feature_w": feature_width,
                            "sampling_seed": sampling_seed,
                            "objects": len(small_objects),
                            "object_mean": object_mean,
                            "background_mean": background_mean,
                            "obac": obac,
                            "cells": cells,
                        }
                    )
    finally:
        for handle in handles:
            handle.remove()

    output.mkdir(parents=True)
    if rows:
        with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = []
    for stride in sorted({row["stride"] for row in rows}):
        subset = [row for row in rows if row["stride"] == stride]
        image_ids = sorted({row["image"] for row in subset})
        image_means = [
            float(np.mean([row["obac"] for row in subset if row["image"] == image_id]))
            for image_id in image_ids
        ]
        summary.append({"stride": stride, "images": len(image_means), "image_mean_obac": float(np.mean(image_means))})
    metadata = {
        "weights": str(Path(args.weights).resolve()),
        "imgsz": args.imgsz,
        "sampling_seeds": args.sampling_seeds,
        "prediction_input_layers": prediction_layers,
        "protocol": "resized max side <32; exclude all annotations and padding; equal cell sampling",
        "warning": "Revised diagnostic; do not compare numerically with legacy private OBAC tables.",
        "summary": summary,
    }
    (output / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
