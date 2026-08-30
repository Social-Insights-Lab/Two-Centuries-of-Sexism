# Scripts

These are the scripts used to build the dataset and produce every number in
the paper, published as they were run. Scripts that only need the released
data files run out of the box:

- `download_data.py`: fetch the classified subset from Hugging Face
- `analysis/paper_stats.py`: reproduce Tables 6, 7, and 9 and the
  chi-squared, logistic regression, and Fisher tests

The remaining scripts are provided for provenance and reproduction from
scratch. They were written against the working research repository, so paths
near the top of each file (input parquet/jsonl locations) may need adjusting
to your local layout:

- `corpus/`: build the gender-matched corpus from raw Hansard HTML and the
  MP database (`create_unified_complete_datasets.py`,
  `create_enhanced_gender_dataset.py`, `mp_matcher_corrected.py`), and
  compute the paper's Table 1 (`dataset_stats.py`)
- `extraction/extract_suffrage_reliable.py`: two-tier keyword extraction of
  the 6,531 women's rights speeches
- `classification/`: build context windows (`05_build_corpus_with_context.py`),
  run the two-pass LLM classification (`03_run_v8_llm.py`, requires
  `HANSARD_ANTHROPIC_API_KEY`), and the cross-model runs
  (`06_run_cross_llm.py`, requires `OPENROUTER_HANSARD_API_KEY`)
- `annotation/`: validation sampling, the Streamlit annotation app, the
  agreement analysis, and the disagreement-resolution app
- `analysis/`: gold-label construction, LLM-vs-gold metrics, baselines,
  sentiment confound, noise induction, and the rebuttal-era robustness
  checks (irrelevant-rate distribution, speaker-level aggregation)
- `figures/`: Figure 1 (stance over time)
- `upload_to_hf.py`: one-time dataset upload (maintainers only)
