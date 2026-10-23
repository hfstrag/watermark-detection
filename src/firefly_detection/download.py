"""Download a subset (e.g. the Firefly folder) of the Synthbuster dataset.

Synthbuster (Bammey, 2023) is distributed on Zenodo as a single ~12.4 GB zip
containing all generators. Zenodo supports HTTP range requests, so we read the
zip's central directory remotely and only fetch the members we need.

Usage:
    firefly-download --list            # show folders in the archive
    firefly-download --folder firefly  # download the Firefly images
"""

import argparse
import io
import os
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from firefly_detection.config import DATA_DIR

SYNTHBUSTER_URL = "https://zenodo.org/records/10066460/files/synthbuster.zip?download=1"


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file object backed by HTTP range requests."""

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Range"].split("/")[-1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_remote_zip(url: str = SYNTHBUSTER_URL) -> zipfile.ZipFile:
    # Buffer reads so zipfile's small header reads don't each become a request.
    return zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 20))


def folder_counts(zf: zipfile.ZipFile) -> Counter:
    """Number of files per directory in the archive."""
    return Counter(os.path.dirname(i.filename) for i in zf.infolist() if not i.is_dir())


def extract_folder(zf: zipfile.ZipFile, folder: str, out: Path, verbose: bool = True) -> int:
    """Extract every file under ``folder`` into ``out``; skip files already complete.

    Returns the number of matching members.
    """
    members = [i for i in zf.infolist() if not i.is_dir() and f"/{folder}/" in f"/{i.filename}"]
    if not members:
        raise ValueError(f"No members matching folder '{folder}'")
    if verbose:
        total = sum(i.file_size for i in members)
        print(f"{len(members)} files, {total / 1e6:.1f} MB")

    for k, info in enumerate(members, 1):
        dest = Path(out) / info.filename
        if dest.exists() and dest.stat().st_size == info.file_size:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        with zf.open(info) as src, open(part, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
        os.replace(part, dest)
        if verbose and (k % 50 == 0 or k == len(members)):
            print(f"  {k}/{len(members)}", flush=True)
    return len(members)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--folder", default="firefly", help="generator folder to extract")
    p.add_argument("--out", type=Path, default=DATA_DIR / "synthbuster", help="output dir")
    p.add_argument("--list", action="store_true", help="only list folders in the archive")
    args = p.parse_args(argv)

    zf = open_remote_zip()
    if args.list:
        for folder, n in sorted(folder_counts(zf).items()):
            print(f"{n:6d}  {folder}")
        return
    try:
        extract_folder(zf, args.folder, args.out)
    except ValueError as e:
        raise SystemExit(f"{e}. Use --list.") from e


if __name__ == "__main__":
    main()
