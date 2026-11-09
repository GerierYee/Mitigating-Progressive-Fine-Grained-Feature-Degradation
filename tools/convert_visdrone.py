"""Convert official VisDrone2019-DET annotations to YOLO detection format."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image


SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}


def convert_split(source: Path, output: Path, split: str) -> None:
    raw_root = source / SPLITS[split]
    image_source = raw_root / "images"
    annotation_source = raw_root / "annotations"
    image_output = output / "images" / split
    label_output = output / "labels" / split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    converted = 0
    for image_path in sorted(image_source.glob("*.jpg")):
        shutil.copy2(image_path, image_output / image_path.name)
        annotation_path = annotation_source / f"{image_path.stem}.txt"
        labels = []
        if annotation_path.is_file():
            with Image.open(image_path) as image:
                image_width, image_height = image.size
            for line in annotation_path.read_text(encoding="utf-8").splitlines():
                values = line.split(",")
                if len(values) < 8:
                    continue
                left, top, width, height = map(float, values[:4])
                score, category = int(float(values[4])), int(float(values[5]))
                if score <= 0 or not 1 <= category <= 10 or min(width, height) <= 0:
                    continue
                center_x = (left + width / 2) / image_width
                center_y = (top + height / 2) / image_height
                labels.append(
                    f"{category - 1} {center_x:.8f} {center_y:.8f} "
                    f"{width / image_width:.8f} {height / image_height:.8f}"
                )
        (label_output / f"{image_path.stem}.txt").write_text("\n".join(labels), encoding="utf-8")
        converted += 1
    print(f"{split}: converted {converted} images")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Directory containing the official VisDrone split folders")
    parser.add_argument("--output", required=True, help="Destination dataset root")
    parser.add_argument("--splits", nargs="+", choices=tuple(SPLITS), default=("train", "val", "test"))
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    for split in args.splits:
        split_root = source / SPLITS[split]
        if not split_root.is_dir():
            raise FileNotFoundError(split_root)
        convert_split(source, output, split)


if __name__ == "__main__":
    main()
