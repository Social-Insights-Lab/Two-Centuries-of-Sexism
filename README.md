# Two Centuries of Sexism in British Parliament

Data and code for the EMNLP 2026 paper "Two Centuries of Sexism in British
Parliament: A Computational Analysis of Women's Representation in the Hansard
Corpus" (Khursheed, Sawkar, and KhudaBukhsh).

We analyse 6,531 speeches on women's suffrage and political representation
drawn from 200 years of UK parliamentary debate (Hansard, 1803-2005),
classifying stance and ambivalent sexism (Glick and Fiske, 1996) with LLMs
validated against expert annotation. We also release the underlying corpus: an
organized, gender-matched version of the Hansard Corpus with 6.78 million
speeches across 1.2 million debates.

## The datasets

Large files are hosted on Hugging Face; this repository holds code,
prompts, human annotations, and result files.

### 1. Hansard gender-matched corpus (Hugging Face)

`https://huggingface.co/datasets/omarkhursheed/hansard-gendered-corpus`

- 6,783,015 speeches across 1,197,828 debates, 1803-2005, both chambers
- Speaker metadata: canonical name, person id, party, constituency, gender
- 89.3% gender-match rate for House of Commons speeches
- Per-year parquet files (201 speech files + 201 debate files, ~9.4 GB)
- MP gender database (`house_members_gendered.parquet`)

See [DATASHEET.md](DATASHEET.md) for the full schema and known limitations.

### 2. Women's political rights subset (Hugging Face, same repo)

- `corpus_with_context.parquet`: the 6,531 keyword-extracted speeches, each
  with up to 5 preceding and 5 following speeches as context
- `speech_classifications.parquet`: one row per speech with speaker
  metadata, LLM stance labels (for / against / both / irrelevant), hostile and
  benevolent sexism flags with subcategories, rationales, and supporting
  verbatim quotes

### 3. In this repository

- `prompts/`: the exact two-pass classification prompts (stance, sexism)
- `data/annotations/`: independent labels from both annotators, consensus
  resolutions, and the merged gold labels for the 300-speech validation set
- `data/classifications/`: raw per-speech LLM outputs (stance and sexism)
- `data/validation/validation_sample.parquet`: the 300-speech validation set
- `data/results/`: baselines, cross-model runs (GPT-5, Gemini 2.5 Flash,
  DeepSeek V3), sentiment confound, noise induction, agreement statistics

## Quickstart

```python
from datasets import load_dataset

# Classified women's rights speeches
ds = load_dataset("omarkhursheed/hansard-gendered-corpus",
                  data_files="womens_rights/speech_classifications.parquet")

# One year of the full corpus
speeches_1918 = load_dataset("omarkhursheed/hansard-gendered-corpus",
                             data_files="corpus/speeches/speeches_1918.parquet")
```

Or fetch the files into `data/`, where the scripts in this repo expect them:

```bash
pip install huggingface_hub

# Women's rights subset (~320 MB) -- enough for all analysis scripts
python scripts/download_data.py

# Subset plus the full 6.78M-speech corpus (~9.4 GB, into data/corpus/)
python scripts/download_data.py --full
```

Equivalent CLI: `hf download omarkhursheed/hansard-gendered-corpus
--repo-type dataset --local-dir data --include "corpus/*" "womens_rights/*"`

## Reproducing the paper

| Paper element | Script |
|---|---|
| Table 1 (dataset statistics) | `scripts/corpus/dataset_stats.py` |
| Keyword extraction (Appendix A) | `scripts/extraction/extract_suffrage_reliable.py` |
| Gender matching (Appendix B) | `scripts/corpus/` |
| LLM classification (Appendix C) | `scripts/classification/run_llm_classification.py` |
| Validation sampling and annotation apps | `scripts/annotation/` |
| Gold labels | `scripts/analysis/build_gold.py` |
| Inter-annotator agreement (Table 4 human row, sexism table) | `scripts/annotation/agreement.py` |
| LLM vs gold: Tables 3, 5, 8, sexism validation | `scripts/analysis/validation_metrics.py` |
| Baselines (Table 4) | `scripts/analysis/baselines.py` |
| Cross-model runs (Tables 5, 8) | `scripts/classification/run_cross_llm.py` |
| Tables 6, 7, 9; chi-squared, logistic regression, Fisher tests | `scripts/analysis/paper_stats.py` |
| Sentiment confound (Appendix E) | `scripts/analysis/sentiment_confound.py` |
| Noise induction (Appendix H) | `scripts/analysis/noise_induction.py` |
| Irrelevant-rate and speaker aggregation (Appendices K, L) | `scripts/analysis/irrelevant_by_time.py`, `speaker_aggregation.py` |
| Figure 1 | `scripts/figures/stance_temporal.py` |

Classification scripts require API keys via environment variables
(`HANSARD_ANTHROPIC_API_KEY`, `OPENROUTER_HANSARD_API_KEY`); analysis scripts
run from the released data alone.

## Licensing

- Code: MIT (see [LICENSE](LICENSE))
- Hansard speech text: [Open Parliament Licence v3.0](https://www.parliament.uk/site-information/copyright-parliament/open-parliament-licence/)
- Annotations, labels, and derived metadata: CC BY 4.0

## Citation

```bibtex
@inproceedings{khursheed2026twocenturies,
  title     = {Two Centuries of Sexism in British Parliament: A Computational
               Analysis of Women's Representation in the Hansard Corpus},
  author    = {Khursheed, Mohammad Omar and Sawkar, Mandira and
               KhudaBukhsh, Ashiqur R.},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in
               Natural Language Processing},
  year      = {2026}
}
```
