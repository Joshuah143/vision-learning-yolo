#!/usr/bin/env python3
"""Utility to visualize YOLO annotations before training."""

from __future__ import annotations

import argparse
import ast
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

try:
    from gen_dataset import DEFAULT_DATASET_ROOT
except ImportError:  # pragma: no cover - fallback if script used standalone
    DEFAULT_DATASET_ROOT = Path("fizz_yolo_dataset")

LOGGER = logging.getLogger("visualize_labels")


VALID_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True, slots=True)
class YoloAnnotation:
    """Single YOLO-format box."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def as_pixels(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        """Convert normalized coordinates to pixel bounds (xmin, ymin, xmax, ymax)."""
        x_c = self.x_center * image_width
        y_c = self.y_center * image_height
        w = self.width * image_width
        h = self.height * image_height

        xmin = int(round(x_c - w / 2.0))
        ymin = int(round(y_c - h / 2.0))
        xmax = int(round(x_c + w / 2.0))
        ymax = int(round(y_c + h / 2.0))

        return xmin, ymin, xmax, ymax


def load_names_from_yaml(yaml_path: Path) -> list[str]:
    """Extract class names from fizz_dataset.yaml without depending on PyYAML."""
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found at {yaml_path}")

    text = yaml_path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        yaml = None

    if yaml is not None:
        data = yaml.safe_load(text)
        names = data.get("names")
        if isinstance(names, list):
            return [str(name) for name in names]

    # Fallback parser: simple extraction of the names list
    for line in text.splitlines():
        if line.strip().startswith("names:"):
            _, value = line.split(":", 1)
            value = value.strip()
            parsed = ast.literal_eval(value)
            if not isinstance(parsed, Sequence):
                raise ValueError(f"Unable to parse names from {yaml_path}")
            return [str(name) for name in parsed]

    raise ValueError(f"No 'names' entry found in {yaml_path}")


def parse_label_file(path: Path) -> list[YoloAnnotation]:
    """Parse a YOLO label file."""
    annotations: list[YoloAnnotation] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            LOGGER.warning("Skipping malformed line %d in %s", line_no, path)
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError as exc:
            LOGGER.warning("Skipping unparsable line %d in %s: %s", line_no, path, exc)
            continue

        annotations.append(
            YoloAnnotation(
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )
    return annotations


def pick_color(class_id: int) -> tuple[int, int, int]:
    """Generate a deterministic BGR color for a given class id."""
    rng = np.random.default_rng(class_id * 9973)
    color = rng.integers(64, 256, size=3, dtype=np.int16)
    return int(color[0]), int(color[1]), int(color[2])


def find_image_path(images_dir: Path, stem: str) -> Path | None:
    """Locate the image corresponding to a label stem."""
    for suffix in VALID_IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def annotate_image(
    image: np.ndarray,
    annotations: Sequence[YoloAnnotation],
    class_names: Sequence[str],
    line_thickness: int,
    font_scale: float,
) -> np.ndarray:
    """Draw bounding boxes and class labels on a copy of the image."""
    annotated = image.copy()
    image_height, image_width = annotated.shape[:2]

    for ann in annotations:
        xmin, ymin, xmax, ymax = ann.as_pixels(image_width, image_height)
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(image_width - 1, xmax)
        ymax = min(image_height - 1, ymax)

        if xmin >= xmax or ymin >= ymax:
            LOGGER.debug("Skipping degenerate bbox for class %d", ann.class_id)
            continue

        color = pick_color(ann.class_id)
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, line_thickness)

        class_name = class_names[ann.class_id] if 0 <= ann.class_id < len(class_names) else f"id_{ann.class_id}"
        label = f"{class_name}"

        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
        )
        text_x = xmin
        text_y = max(0, ymin - 5)
        rect_top = max(0, text_y - text_height - baseline)
        rect_bottom = text_y + baseline
        cv2.rectangle(
            annotated,
            (text_x, rect_top),
            (text_x + text_width, rect_bottom),
            color,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            line_thickness,
            lineType=cv2.LINE_AA,
        )

    return annotated


def iter_label_files(labels_dir: Path, shuffle: bool, seed: int | None) -> Iterable[Path]:
    """Yield label files, optionally shuffled."""
    paths = sorted(labels_dir.glob("*.txt"))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(paths)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize YOLO labeled data before training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Directory containing images/, labels/, and fizz_dataset.yaml.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=16,
        help="Maximum number of labeled images to render.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the order of samples before visualization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed controlling shuffle order (only used with --shuffle).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to save annotated images. "
        "Defaults to <dataset_root>/visualizations/<split>.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated images with cv2.imshow (requires GUI).",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=2,
        help="Bounding box line thickness.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.6,
        help="Font scale for class labels.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    dataset_root = args.dataset_root.resolve()
    images_dir = dataset_root / "images" / args.split
    labels_dir = dataset_root / "labels" / args.split
    yaml_path = dataset_root / "fizz_dataset.yaml"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Expected directories {images_dir} and {labels_dir} to exist."
        )

    class_names = load_names_from_yaml(yaml_path)
    LOGGER.info("Loaded %d class names", len(class_names))

    if args.output_dir is None and not args.show:
        args.output_dir = dataset_root / "visualizations" / args.split

    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Saving annotated images to %s", args.output_dir)

    label_paths = list(iter_label_files(labels_dir, args.shuffle, args.seed))
    if not label_paths:
        LOGGER.warning("No label files found in %s", labels_dir)
        return

    processed = 0
    for label_path in label_paths:
        if args.max_images is not None and processed >= args.max_images:
            break

        annotations = parse_label_file(label_path)
        if not annotations:
            LOGGER.debug("No annotations in %s", label_path)
            continue

        stem = label_path.stem
        image_path = find_image_path(images_dir, stem)
        if image_path is None:
            LOGGER.warning("No image found for label %s", label_path)
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            LOGGER.warning("Failed to read image %s", image_path)
            continue

        annotated = annotate_image(
            image,
            annotations,
            class_names,
            line_thickness=max(1, args.line_thickness),
            font_scale=max(0.1, args.font_scale),
        )

        if args.output_dir is not None:
            output_path = args.output_dir / f"{stem}_viz.jpg"
            if not cv2.imwrite(str(output_path), annotated):
                LOGGER.error("Failed to write annotated image %s", output_path)

        if args.show:
            window_name = f"YOLO Visualization - {stem}"
            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(0)
            cv2.destroyWindow(window_name)
            if key & 0xFF in (ord("q"), 27):  # q or ESC
                LOGGER.info("Stopping visualization on user request.")
                break

        processed += 1

    if args.show:
        cv2.destroyAllWindows()

    LOGGER.info("Annotated %d images.", processed)


if __name__ == "__main__":
    main()

