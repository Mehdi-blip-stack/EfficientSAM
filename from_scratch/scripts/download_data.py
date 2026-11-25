"""Download the Oxford-IIIT Pets archives and prebuild requested caches."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tinysam.data import build_cache, download_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "data"))
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    download_data(Path(args.root) / "raw")
    for split in ("train", "val", "test"):
        path = build_cache(split, args.size, args.root)
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
