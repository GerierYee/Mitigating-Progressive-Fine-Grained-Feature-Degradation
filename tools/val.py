"""Validate a released checkpoint with the repository evaluation settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/data/visdrone.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", default="runs/val/verified_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = YOLO(args.weights).val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=0.001,
        iou=0.7,
        max_det=1000,
        rect=False,
        half=False,
        plots=False,
        save_json=True,
    )
    indices = [int(index) for index in result.box.ap_class_index]
    per_class = [
        {
            "class_id": class_id,
            "name": result.names[class_id],
            "ap50": float(values[0]),
            "ap50_95": float(sum(values) / len(values)),
        }
        for class_id, values in zip(indices, result.box.all_ap.tolist())
    ]
    payload = {
        "weights": str(Path(args.weights).resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": 0.001,
        "nms_iou": 0.7,
        "max_det": 1000,
        "precision": "FP32",
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
        "per_class": per_class,
        "metric_note": "Ultralytics evaluation on converted YOLO labels; not the official challenge server metric.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
