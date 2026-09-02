"""Project-anchored paths — everything lives under this repo, no sibling-project
dependencies (checkpoints and the ChEMBL CSV are vendored in ``checkpoints/`` and
``data/external/``)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_EXTERNAL = DATA_DIR / "external"
DATA_INTERIM = DATA_DIR / "interim"

RESULTS_DIR = PROJECT_ROOT / "results"

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
REFERENCE_DIR = PROJECT_ROOT / "reference"


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_INTERIM, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
