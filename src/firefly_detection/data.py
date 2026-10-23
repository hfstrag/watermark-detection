"""Loading utilities for the Synthbuster images."""

from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from firefly_detection.config import SYNTHBUSTER_DIR

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
# ITU-R BT.601 luma weights, applied in float to avoid rounding to 8 bits.
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114])


def list_images(generator: str = "firefly", root: Path = SYNTHBUSTER_DIR) -> list[Path]:
    """Sorted list of image paths for one Synthbuster generator folder."""
    folder = Path(root) / generator
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} not found; run `firefly-download` first")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_image(path: Path, gray: bool = False, dtype=np.float32) -> np.ndarray:
    """Load an image as an (H, W, 3) or (H, W) array with values in [0, 255].

    With ``gray=True`` the luminance is computed in floating point, so it keeps
    sub-integer detail instead of being re-quantized to 8 bits.
    """
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.float64)
    return (rgb @ LUMA_WEIGHTS if gray else rgb).astype(dtype)


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) read from the header, without decoding the pixels."""
    with Image.open(path) as img:
        return img.size


def group_by_size(paths: Iterable[Path]) -> dict[tuple[int, int], list[Path]]:
    """Group image paths by (width, height), largest group first."""
    groups = defaultdict(list)
    for p in paths:
        groups[image_size(p)].append(p)
    return dict(sorted(groups.items(), key=lambda kv: -len(kv[1])))


def iter_images(
    generator: str = "firefly",
    root: Path = SYNTHBUSTER_DIR,
    gray: bool = False,
    limit: int | None = None,
) -> Iterator[tuple[Path, np.ndarray]]:
    """Yield (path, array) pairs lazily, so the whole set needn't fit in memory."""
    for path in list_images(generator, root)[:limit]:
        yield path, load_image(path, gray=gray)
