"""Loading utilities for the Synthbuster images."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from firefly_detection.config import SYNTHBUSTER_DIR

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def list_images(generator: str = "firefly", root: Path = SYNTHBUSTER_DIR) -> list[Path]:
    """Sorted list of image paths for one Synthbuster generator folder."""
    folder = Path(root) / generator
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} not found; run `firefly-download` first")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_image(path: Path, gray: bool = False, dtype=np.float32) -> np.ndarray:
    """Load an image as an (H, W, 3) or (H, W) array with values in [0, 255]."""
    with Image.open(path) as img:
        return np.asarray(img.convert("L" if gray else "RGB"), dtype=dtype)


def iter_images(
    generator: str = "firefly",
    root: Path = SYNTHBUSTER_DIR,
    gray: bool = False,
    limit: int | None = None,
) -> Iterator[tuple[Path, np.ndarray]]:
    """Yield (path, array) pairs lazily, so the whole set needn't fit in memory."""
    for path in list_images(generator, root)[:limit]:
        yield path, load_image(path, gray=gray)
