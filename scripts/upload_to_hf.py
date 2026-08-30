"""One-time upload of the dataset to Hugging Face.

Uploads the dataset card and womens_rights files from hf_staging/, and the
full corpus directly from the external drive (not copied locally first).

Prereqs:
  uv tool install "huggingface_hub[cli]" && hf auth login
  External drive mounted at /Volumes/safety_project_backups

Usage:
  python scripts/upload_to_hf.py --repo-id HF_USER/hansard-gendered-corpus
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

STAGING = Path(__file__).resolve().parents[1] / "hf_staging"
DRIVE = Path("/Volumes/safety_project_backups/hansard-nlp-explorer/data-hansard")
SPEECHES = DRIVE / "derived_v2" / "speeches_complete"
DEBATES = DRIVE / "derived_v2" / "debates_complete"
MP_DB = DRIVE / "house_members_gendered_updated.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--private", action="store_true",
                    help="Create as private (flip to public on HF later)")
    args = ap.parse_args()

    for p in (SPEECHES, DEBATES, MP_DB):
        assert p.exists(), f"Missing: {p} (is the drive mounted?)"

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset",
                    private=args.private, exist_ok=True)

    api.upload_file(path_or_fileobj=STAGING / "README.md",
                    path_in_repo="README.md",
                    repo_id=args.repo_id, repo_type="dataset")
    api.upload_folder(folder_path=STAGING / "womens_rights",
                      path_in_repo="womens_rights",
                      repo_id=args.repo_id, repo_type="dataset")
    api.upload_file(path_or_fileobj=MP_DB,
                    path_in_repo="corpus/house_members_gendered.parquet",
                    repo_id=args.repo_id, repo_type="dataset")

    # upload_folder is multi-commit and resumable for large folders
    api.upload_folder(folder_path=SPEECHES,
                      path_in_repo="corpus/speeches",
                      repo_id=args.repo_id, repo_type="dataset",
                      allow_patterns="*.parquet")
    api.upload_folder(folder_path=DEBATES,
                      path_in_repo="corpus/debates",
                      repo_id=args.repo_id, repo_type="dataset",
                      allow_patterns="*.parquet")
    print("Upload complete.")


if __name__ == "__main__":
    main()
