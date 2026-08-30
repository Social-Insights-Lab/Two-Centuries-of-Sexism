---
license: other
license_name: open-parliament-licence-3.0
license_link: https://www.parliament.uk/site-information/copyright-parliament/open-parliament-licence/
language:
- en
task_categories:
- text-classification
tags:
- parliamentary-debates
- political-science
- computational-social-science
- gender
- sexism-detection
- stance-detection
- hansard
size_categories:
- 1K<n<10K
pretty_name: Two Centuries of Sexism - Classified Hansard Speeches
---

# Two Centuries of Sexism: Classified Hansard Speeches

The analysis dataset of the EMNLP 2026 paper "Two Centuries of Sexism in
British Parliament: A Computational Analysis of Women's Representation in the
Hansard Corpus" (Khursheed, Sawkar, and KhudaBukhsh): 6,531 UK parliamentary
speeches (1809-2004) on women's suffrage and political representation,
classified for stance and ambivalent sexism (Glick and Fiske, 1996) by
Claude Sonnet 4.6 and validated against expert annotation on a 300-speech
gold set.

The speeches were retrieved from the Hansard corpus with a two-tier keyword
extractor; the extractor, the classification prompts, the human annotations,
and all validation code are in the companion repository:
https://github.com/Social-Insights-Lab/Two-Centuries-of-Sexism

The full 6.78M-speech gender-matched corpus is released separately:
https://huggingface.co/datasets/omarkhursheed/hansard-gendered-corpus

## Contents

```
speech_classifications.parquet   # 6,531 rows: speech text, speaker metadata,
                                 # LLM stance + sexism labels with rationales
                                 # and supporting quotes
corpus_with_context.parquet      # the same speeches with up to 5 preceding
                                 # and 5 following speeches as context
                                 # (what the LLM and annotators saw)
```

`speech_id` is an opaque identifier (s0001-s6531) linking the two files and
the annotation/label files in the companion repository. It is not a key into
the separately released corpus; to re-derive this dataset from the corpus,
use the released extraction pipeline.

## Fields (speech_classifications.parquet)

| Field | Description |
|---|---|
| speech_id | Opaque per-speech identifier (s0001-s6531) |
| speaker / canonical_name | Speaker as printed / resolved MP name |
| gender, party, chamber | Speaker metadata from the MP-matching pipeline |
| year, decade, era, date | Date of the speech |
| word_count, target_text, context_text | Speech text and formatted context |
| llm_stance | for / against / both / irrelevant |
| llm_stance_confidence, llm_stance_rationale | Model confidence and rationale |
| llm_hostile, llm_hostile_subcategories, llm_hostile_quote | Hostile sexism flag, subtypes, verbatim supporting quote (flag null for 14 speeches with schema-incomplete model output; treat as False) |
| llm_benevolent, llm_benevolent_subcategories, llm_benevolent_quote | Benevolent sexism flag, subtypes, quote (same null caveat) |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("omarkhursheed/two-centuries-of-sexism",
                  data_files="speech_classifications.parquet")
```

## Validation

Stance: Claude Sonnet 4.6 reaches Cohen's kappa 0.711 against human consensus
labels on the 300-speech gold set (human-human kappa 0.644). Sexism flags:
precision 0.77-0.86, recall 0.43-0.46 vs human labels. Full validation
detail, cross-model runs (GPT-5, Gemini 2.5 Flash, DeepSeek V3), baselines,
and robustness checks are reported in the paper and reproducible from the
companion repository.

## Licensing

Hansard speech text is Crown/Parliamentary material under the Open
Parliament Licence v3.0. Labels, annotations, and derived metadata are
CC BY 4.0.

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
