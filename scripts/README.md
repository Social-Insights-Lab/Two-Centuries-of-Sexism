# Scripts

All paths are repo-relative; each script's docstring states what it reads and
writes. First fetch the data files the scripts expect:

```bash
python scripts/download_data.py          # women's rights subset (~320 MB)
python scripts/download_data.py --full   # + full corpus (~9.4 GB)
```

## Reproduce the paper's numbers

| Script | Produces |
|---|---|
| `corpus/dataset_stats.py` | Table 1 (needs `--full` corpus) |
| `analysis/paper_stats.py` | Tables 6, 7, 9; chi-squared, logistic regression, Fisher tests |
| `analysis/validation_metrics.py` | Tables 3, 5, 8; sexism validation metrics |
| `annotation/agreement.py` | Inter-annotator agreement (Table 4 human row) |
| `analysis/build_gold.py` | Gold labels from the two annotators + consensus |
| `analysis/baselines.py` | Table 4 baseline rows |
| `analysis/sentiment_confound.py` | Appendix E (downloads DistilBERT-SST-2) |
| `analysis/noise_induction.py` | Appendix H |
| `analysis/irrelevant_by_time.py` | Appendix K |
| `analysis/speaker_aggregation.py` | Appendix L |
| `figures/stance_temporal.py` | Figure 1 |

## Rebuild the analysis corpus from the full corpus

| Script | Step |
|---|---|
| `extraction/extract_suffrage_reliable.py` | Two-tier keyword extraction (Appendix A) |
| `classification/build_corpus_with_context.py` | Attach 5-before/5-after context windows |

## Re-run the LLM classification (needs API keys)

| Script | Key |
|---|---|
| `classification/run_llm_classification.py` | `HANSARD_ANTHROPIC_API_KEY` |
| `classification/run_cross_llm.py` (GPT-5 / Gemini / DeepSeek) | `OPENROUTER_HANSARD_API_KEY` |

## Annotation tooling

| Script | Purpose |
|---|---|
| `annotation/annotation_app.py` | Streamlit app the annotators used |
| `annotation/resolution_app.py` | Streamlit app for resolving disagreements |
| `annotation/create_validation_sample.py` | Sampling procedure for the validation set (output shipped as `data/validation/validation_sample.parquet`) |

## Corpus construction from raw Hansard

The full corpus was built from a crawl of the Historic Hansard website
(https://api.parliament.uk/historic-hansard/). The crawl itself (~19 GB of
HTML) is not redistributed; these scripts document the pipeline from crawl
to the released corpus:

| Script | Step |
|---|---|
| `corpus/create_enhanced_gender_dataset.py` | Parse debates, extract speech turns |
| `corpus/mp_matcher_corrected.py` | Match speakers to the MP database |
| `corpus/create_unified_complete_datasets.py` | Assemble the per-year speech/debate parquet files |

`upload_to_hf.py` is the one-time dataset upload used by the maintainers.
