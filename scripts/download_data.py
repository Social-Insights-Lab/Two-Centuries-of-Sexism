"""Download the datasets from Hugging Face into data/.

By default fetches the classified women's rights dataset (~320 MB), which is
all the analysis scripts need. Pass --full to also download the full
6.78M-speech corpus (~9.4 GB) into data/corpus/.

Usage:
  python scripts/download_data.py           # classified dataset only
  python scripts/download_data.py --full    # + full corpus

Requires: huggingface_hub (pip install huggingface_hub)
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

CLASSIFIED_REPO = "omarkhursheed/two-centuries-of-sexism"
CORPUS_REPO = "omarkhursheed/hansard-gendered-corpus"
DATA = Path(__file__).resolve().parents[1] / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also download the full corpus (~9.4 GB)")
    args = ap.parse_args()

    snapshot_download(repo_id=CLASSIFIED_REPO, repo_type="dataset",
                      local_dir=DATA / "womens_rights",
                      allow_patterns="*.parquet")
    print(f"Classified dataset -> {DATA / 'womens_rights'}")

    if args.full:
        snapshot_download(repo_id=CORPUS_REPO, repo_type="dataset",
                          local_dir=DATA, allow_patterns="corpus/*")
        print(f"Full corpus -> {DATA / 'corpus'}")


if __name__ == "__main__":
    main()
