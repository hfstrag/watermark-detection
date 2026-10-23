import zipfile

import pytest

from firefly_detection.download import extract_folder, folder_counts


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "synthbuster.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("synthbuster/firefly/a.png", b"aaa")
        zf.writestr("synthbuster/firefly/b.png", b"bbbb")
        zf.writestr("synthbuster/dalle3/c.png", b"c")
        zf.writestr("synthbuster/prompts.csv", b"x")
    return zipfile.ZipFile(path)


def test_folder_counts(archive):
    counts = folder_counts(archive)
    assert counts["synthbuster/firefly"] == 2
    assert counts["synthbuster/dalle3"] == 1


def test_extract_folder_only_matching(archive, tmp_path):
    out = tmp_path / "out"
    assert extract_folder(archive, "firefly", out, verbose=False) == 2
    files = sorted(p.name for p in out.rglob("*") if p.is_file())
    assert files == ["a.png", "b.png"]
    assert (out / "synthbuster/firefly/b.png").read_bytes() == b"bbbb"


def test_extract_folder_unknown(archive, tmp_path):
    with pytest.raises(ValueError):
        extract_folder(archive, "midjourney", tmp_path)
