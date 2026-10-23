# Firefly detection

Exploiting the upsampling of the Adobe Firefly image generator for manipulation detection.

Firefly images appear to carry a periodic pattern, robust to processing such as JPEG
compression, which is believed to come from an upsampling step during generation. This
project estimates that shared signal from a dataset of Firefly images and derives a
statistical test for its presence at image and patch level, following:

1. J. Butora and P. Bas, *Detection of the Adobe Pattern*, EUSIPCO 2024.
2. J. Butora and P. Bas, *The Adobe Hidden Feature and its Impact on Sensor Attribution*,
   IH&MMSec 2024.

See `docs/` for the subject description and the reference papers.

## Setup

Requires Python ≥ 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` elsewhere
python -m pip install -e ".[dev]"
```

## Data

Firefly images come from [Synthbuster](https://zenodo.org/records/10066460)
(Bammey, 2023): 1000 lossless RGB PNGs, ~1.55 GB. The downloader fetches only the
requested generator folder from the 12.4 GB archive using HTTP range requests:

```bash
firefly-download --list             # folders available in the archive
firefly-download --folder firefly   # -> data/synthbuster/synthbuster/firefly/
```

Other generators (`dalle3`, `stable-diffusion-xl`, ...) can be fetched the same way for
false-positive evaluation. Set `FIREFLY_DATA_DIR` to store data elsewhere.

```python
from firefly_detection.data import iter_images

for path, img in iter_images("firefly", gray=True, limit=10):
    ...
```

## Layout

```
docs/                    subject and reference papers
src/firefly_detection/   library code
tests/                   pytest suite
notebooks/               exploratory analysis
data/                    datasets (git-ignored)
results/                 figures and outputs (git-ignored)
```

## Development

```bash
pytest
ruff check . && ruff format .
```
