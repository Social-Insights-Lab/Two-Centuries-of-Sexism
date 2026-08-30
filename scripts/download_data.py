"""Download the dataset from Hugging Face into data/.

By default fetches the women's rights subset (~320 MB), which is all the
analysis scripts need. Pass --full to also download the full 6.78M-speech
corpus (~9.4 GB) into data/corpus/.

Usage:
  python scripts/download_data.py           # subset only
  python scripts/download_data.py --full    # subset + full corpus

Requires: huggingface_hub (pip install huggingface_hub)
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "omarkhursheed/hansard-gendered-corpus"
DATA = Path(__file__).resolve().parents[1] / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also download the full corpus (~9.4 GB)")
    args = ap.parse_args()

    patterns = ["womens_rights/*"]
    if args.full:
        patterns.append("corpus/*")

    snapshot_download(repo_id=REPO_ID, repo_type="dataset",
                      local_dir=DATA, allow_patterns=patterns)
    print(f"Downloaded {', '.join(patterns)} -> {DATA}")


if __name__ == "__main__":
    main()
