"""Train CMD-YOLO26 with an explicit, reproducible command line."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="configs/models/cmd_yolo26n_pfgr.yaml")
    parser.add_argument("--data", default="configs/data/visdrone.yaml")
    parser.add_argument("--pretrained", default="", help="Optional checkpoint to partially load before training")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="cmd_yolo26_pfgr_seed0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.model, args.data):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    model = YOLO(args.model)
    if args.pretrained:
        model.load(args.pretrained)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        project=args.project,
        name=args.name,
        exist_ok=False,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
