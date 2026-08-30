"""Download the women's rights subset files from Hugging Face into data/.

The full 9.4 GB corpus is not downloaded by default; see README for how to
load it directly with the datasets library.

Requires: huggingface_hub (pip install huggingface_hub)
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "HF_USER/hansard-gendered-corpus"
DEST = Path(__file__).resolve().parents[1] / "data" / "womens_rights"

FILES = [
    "womens_rights/v8_corpus_classifications.parquet",
    "womens_rights/corpus_with_context.parquet",
]


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        path = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset")
        target = DEST / Path(f).name
        if not target.exists():
            target.symlink_to(path)
        print(f"{f} -> {target}")


if __name__ == "__main__":
    main()
