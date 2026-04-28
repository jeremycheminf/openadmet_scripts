"""Download PXR challenge CSVs from HuggingFace and do a quick sanity check."""
import sys
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

BASE = (
    "https://huggingface.co/datasets/openadmet/pxr-challenge-train-test"
    "/resolve/main/"
)

FILES = {
    "train_raw.csv":         "pxr-challenge_TRAIN.csv",
    "test_raw.csv":          "pxr-challenge_TEST_BLINDED.csv",
    "counter_assay_raw.csv": "pxr-challenge_counter-assay_TRAIN.csv",
}


def download(remote_name: str, local_path: Path, force: bool = False):
    if local_path.exists() and not force:
        print(f"  {local_path.name} already exists — skipping")
        return
    url = BASE + remote_name
    print(f"  GET {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    local_path.write_bytes(r.content)
    print(f"  Saved {local_path.name} ({len(r.content) // 1024} KB)")


def main():
    print("=== Downloading PXR challenge data ===")
    for local_name, remote_name in FILES.items():
        download(remote_name, DATA_DIR / local_name)

    print("\n=== Sanity checks ===")
    train = pd.read_csv(DATA_DIR / "train_raw.csv")
    test  = pd.read_csv(DATA_DIR / "test_raw.csv")

    print(f"Train shape : {train.shape}")
    print(f"Test shape  : {test.shape}")
    print(f"Train cols  : {train.columns.tolist()}")
    print(f"Test cols   : {test.columns.tolist()}")

    print(f"\npEC50 (train) — min={train['pEC50'].min():.2f}  "
          f"max={train['pEC50'].max():.2f}  "
          f"null={train['pEC50'].isna().sum()}")

    # Column names include units in parentheses — find the std.error col
    se_col = [c for c in train.columns if c.startswith("pEC50_std.error")][0]
    print(f"\n{se_col} — min={train[se_col].min():.3f}  "
          f"max={train[se_col].max():.3f}  "
          f"null={train[se_col].isna().sum()}")

    if train.shape[0] < 4000:
        print("WARNING: expected ≥ 4000 training rows", file=sys.stderr)
        sys.exit(1)
    if test.shape[0] < 500:
        print("WARNING: expected ≥ 500 test rows", file=sys.stderr)
        sys.exit(1)

    print("\nSource breakdown (train):")
    print(train["source"].value_counts().to_string())

    print("\nBatch breakdown (train):")
    if "OCNT Batch" in train.columns:
        print(train["OCNT Batch"].value_counts().head(10).to_string())

    print(f"\nTest columns: {test.columns.tolist()}")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
