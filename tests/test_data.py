import numpy as np
import pytest
from PIL import Image

from firefly_detection.data import group_by_size, iter_images, list_images, load_image


@pytest.fixture
def fake_root(tmp_path):
    folder = tmp_path / "firefly"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for name in ["b.png", "a.png"]:
        Image.fromarray(rng.integers(0, 256, (16, 24, 3), dtype=np.uint8)).save(folder / name)
    (folder / "notes.txt").write_text("not an image")
    return tmp_path


def test_list_images_sorted_and_filtered(fake_root):
    assert [p.name for p in list_images(root=fake_root)] == ["a.png", "b.png"]


def test_list_images_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_images("nope", root=tmp_path)


def test_load_image_shapes(fake_root):
    path = list_images(root=fake_root)[0]
    rgb, gray = load_image(path), load_image(path, gray=True)
    assert rgb.shape == (16, 24, 3) and rgb.dtype == np.float32
    assert gray.shape == (16, 24)
    assert 0 <= rgb.min() and rgb.max() <= 255


def test_iter_images_limit(fake_root):
    assert len(list(iter_images(root=fake_root, limit=1))) == 1


def test_group_by_size(tmp_path):
    for name, (w, h) in {"a.png": (8, 4), "b.png": (4, 8), "c.png": (8, 4)}.items():
        Image.new("RGB", (w, h)).save(tmp_path / name)
    groups = group_by_size(sorted(tmp_path.iterdir()))
    assert list(groups) == [(8, 4), (4, 8)]
    assert [p.name for p in groups[(8, 4)]] == ["a.png", "c.png"]
