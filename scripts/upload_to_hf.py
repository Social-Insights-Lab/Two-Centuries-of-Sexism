"""One-time upload of the two datasets to Hugging Face (maintainers only).

Dataset 1 (corpus): the full gender-matched Hansard corpus, uploaded
directly from the external drive.
Dataset 2 (classified): the standalone classified women's rights speeches,
uploaded from hf_staging/.

Prereqs:
  hf auth login with a write-role token
  External drive mounted at /Volumes/safety_project_backups (corpus only)

Usage:
  python scripts/upload_to_hf.py --corpus-repo omarkhursheed/hansard-gendered-corpus \
      --classified-repo omarkhursheed/two-centuries-of-sexism [--private]
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

STAGING = Path(__file__).resolve().parents[1] / "hf_staging"
DRIVE = Path("/Volumes/safety_project_backups/hansard-nlp-explorer/data-hansard")
# derived_complete is the build the paper's Table 1 was computed from.
SPEECHES = DRIVE / "derived_complete" / "speeches_complete"
DEBATES = DRIVE / "derived_complete" / "debates_complete"
MP_DB = DRIVE / "house_members_gendered_updated.parquet"


def upload_corpus(api, repo_id, private):
    for p in (SPEECHES, DEBATES, MP_DB):
        assert p.exists(), f"Missing: {p} (is the drive mounted?)"
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=STAGING / "corpus_card.md",
                    path_in_repo="README.md", repo_id=repo_id, repo_type="dataset")
    api.upload_file(path_or_fileobj=MP_DB,
                    path_in_repo="corpus/house_members_gendered.parquet",
                    repo_id=repo_id, repo_type="dataset")
    api.upload_folder(folder_path=SPEECHES, path_in_repo="corpus/speeches",
                      repo_id=repo_id, repo_type="dataset",
                      allow_patterns="*.parquet")
    api.upload_folder(folder_path=DEBATES, path_in_repo="corpus/debates",
                      repo_id=repo_id, repo_type="dataset",
                      allow_patterns="*.parquet")


def upload_classified(api, repo_id, private):
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=STAGING / "classified_card.md",
                    path_in_repo="README.md", repo_id=repo_id, repo_type="dataset")
    api.upload_folder(folder_path=STAGING / "womens_rights", path_in_repo="",
                      repo_id=repo_id, repo_type="dataset",
                      allow_patterns="*.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-repo")
    ap.add_argument("--classified-repo")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    if args.corpus_repo:
        upload_corpus(api, args.corpus_repo, args.private)
    if args.classified_repo:
        upload_classified(api, args.classified_repo, args.private)
    print("Upload complete.")


if __name__ == "__main__":
    main()
