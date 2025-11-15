#!/usr/bin/env python3
import json
import os
import random
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, get_args

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from pipeline_config import PlatesConfig, load_pipeline_config

ClueType = Literal["SIZE", "VICTIM", "CRIME", "TIME", "PLACE", "MOTIVE", "WEAPON", "BANDIT"]
TYPES = list(get_args(ClueType))

SCRIPT_DIR = Path(__file__).resolve().parent
BANNER_PATH = SCRIPT_DIR / "gen_tools" / "clue_banner.png"
FONT_PATH = SCRIPT_DIR / "gen_tools" / "UbuntuMono-Regular.ttf"

banner_canvas = cv2.imread(str(BANNER_PATH))
if banner_canvas is None:
    raise FileNotFoundError(f"Unable to load banner template at {BANNER_PATH}")

PLATE_HEIGHT = 600
PLATE_WIDTH = 400


def _draw_text_with_boxes(
    draw: ImageDraw.ImageDraw,
    text: str,
    start_pos: tuple[float, float],
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> list[dict[str, object]]:
    """Draw text character-by-character and capture bounding boxes."""
    cursor_x, cursor_y = start_pos
    annotations: list[dict[str, object]] = []

    for char in text:
        if char == " ":
            cursor_x += draw.textlength(" ", font=font)
            continue

        bbox = draw.textbbox((cursor_x, cursor_y), char, font=font)
        draw.text((cursor_x, cursor_y), char, fill=fill, font=font)
        annotations.append(
            {
                "char": char,
                "bbox": [int(round(coord)) for coord in bbox],
            }
        )
        cursor_x += draw.textlength(char, font=font)

    return annotations


def save_banner(clue_type: ClueType, clue_text: str, save_path: Path):
    font_size = 90
    blank_plate_pil = Image.fromarray(banner_canvas)
    draw = ImageDraw.Draw(blank_plate_pil)
    monospace = ImageFont.truetype(FONT_PATH, font_size)
    font_color = (255, 0, 0)

    letter_annotations: list[dict[str, object]] = []

    # TYPE
    letter_annotations.extend(
        _draw_text_with_boxes(draw, clue_type, (250, 30), monospace, font_color)
    )
    # CLUE TEXT
    letter_annotations.extend(
        _draw_text_with_boxes(draw, clue_text, (30, 250), monospace, font_color)
    )

    populated_banner = np.array(blank_plate_pil)
    cv2.imwrite(str(save_path), populated_banner)

    metadata = {
        "clue_type": clue_type,
        "clue_text": clue_text,
        "letters": letter_annotations,
    }
    save_path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def random_clue_text():
    # either sample from the CSV list, or just roll your own
    # single word
    if random.random() < 0.5:
        length = random.randint(3, 8)
        return "".join(random.choice(string.ascii_uppercase) for _ in range(length))
    else:
        # two words
        length1 = random.randint(3, 7)
        length2 = min(11 - length1, random.randint(3, 7))

        w1 = "".join(random.choice(string.ascii_uppercase) for _ in range(length1))
        w2 = "".join(random.choice(string.ascii_uppercase) for _ in range(length2))
        return f"{w1} {w2}"


def _generate_plate(task: tuple[int, str, str], output_dir: Path):
    index, key, value = task
    save_banner(key, value, output_dir / f"{index}_{key}_{value}.png")


def generate_plates(config: PlatesConfig | None = None) -> None:
    if config is None:
        config = load_pipeline_config().plates

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.seed is not None:
        random.seed(config.seed)

    print(f"Generating {config.num_plates} plates into {output_dir}")
    tasks = [(i, random.choice(TYPES), random_clue_text()) for i in range(config.num_plates)]
    max_workers = os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_generate_plate, task, output_dir) for task in tasks]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Generating plates",
            unit="plate",
        ):
            future.result()

if __name__ == "__main__":
    generate_plates()
