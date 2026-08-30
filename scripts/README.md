# Scripts

These scripts build the dataset and produce every number in the paper. Paths
are repo-relative; each script's docstring states exactly what it reads and
writes. Scripts that need the Hugging Face files expect them under
`data/womens_rights/` or `data/corpus/` (run `download_data.py` first).

Runnable from the released data alone:

- `download_data.py`: fetch the women's rights subset from Hugging Face
- `corpus/dataset_stats.py`: Table 1 (needs the full corpus downloaded)
- `extraction/extract_suffrage_reliable.py`: two-tier keyword extraction of
  the analysis corpus from the full corpus
- `analysis/paper_stats.py`: Tables 6, 7, 9; chi-squared, logistic
  regression, and Fisher tests
- `analysis/validation_metrics.py`: LLM-vs-gold metrics (Tables 3, 5, 8 and
  the sexism validation table)
- `analysis/build_gold.py`: merge the two annotators' labels and consensus
  resolutions into gold labels
- `annotation/agreement.py`: inter-annotator agreement
- `analysis/baselines.py`: majority / TF-IDF / DeBERTa baselines (Table 4)
- `analysis/sentiment_confound.py`: Appendix E (downloads DistilBERT-SST-2)
- `analysis/noise_induction.py`: Appendix H
- `analysis/irrelevant_by_time.py`, `analysis/speaker_aggregation.py`:
  Appendices K and L
- `figures/stance_temporal.py`: Figure 1
- `annotation/annotation_app.py`, `annotation/resolution_app.py`: the
  Streamlit apps used for annotation and disagreement resolution

Require API keys (re-running the classification):

- `classification/run_llm_classification.py`: two-pass Claude classification
  (`HANSARD_ANTHROPIC_API_KEY`)
- `classification/run_cross_llm.py`: GPT-5 / Gemini / DeepSeek via OpenRouter
  (`OPENROUTER_HANSARD_API_KEY`)

Provenance only (document how released artifacts were built; run against
internal working files not included in the release):

- `corpus/create_unified_complete_datasets.py`,
  `corpus/create_enhanced_gender_dataset.py`, `corpus/mp_matcher_corrected.py`:
  the raw-crawl-to-corpus pipeline and MP gender matching
- `classification/build_corpus_with_context.py`: context-window construction
  (output released as `corpus_with_context.parquet`)
- `annotation/create_validation_sample.py`: validation sampling (output
  released as `data/validation/validation_sample.parquet`)
- `upload_to_hf.py`: one-time dataset upload (maintainers only)
