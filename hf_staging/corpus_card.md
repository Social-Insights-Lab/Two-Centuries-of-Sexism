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
- historical-text
- hansard
size_categories:
- 1M<n<10M
pretty_name: Hansard Gender-Matched Corpus (1803-2005)
---

# Hansard Gender-Matched Corpus (1803-2005)

An organized, speaker-enriched version of the Historic Hansard corpus of UK
parliamentary debates: 6,783,015 speeches across 1,197,828 debates from 203
years (1803-2005) of House of Commons and House of Lords proceedings, with
89.3% speaker gender-matching coverage in the Commons.

Released with the EMNLP 2026 paper "Two Centuries of Sexism in British
Parliament: A Computational Analysis of Women's Representation in the Hansard
Corpus" (Khursheed, Sawkar, and KhudaBukhsh). Code, the keyword extractor,
and the full datasheet are in the companion repository:
https://github.com/Social-Insights-Lab/Two-Centuries-of-Sexism

The paper's classified dataset of 6,531 speeches on women's political rights
is released separately:
https://huggingface.co/datasets/omarkhursheed/two-centuries-of-sexism

## Contents

```
corpus/
  speeches/speeches_YYYY.parquet   # 201 files, one per year: speech text +
                                   # speaker metadata (name, MP id, gender,
                                   # party, constituency)
  debates/debates_YYYY.parquet     # 201 files: debate-level metadata and
                                   # speaker composition
  house_members_gendered.parquet   # unified MP database with gender labels
```

## Usage

```python
from datasets import load_dataset

# One year of the corpus
y1918 = load_dataset("omarkhursheed/hansard-gendered-corpus",
                     data_files="corpus/speeches/speeches_1918.parquet")

# A range of years
franchise_era = load_dataset(
    "omarkhursheed/hansard-gendered-corpus",
    data_files=[f"corpus/speeches/speeches_{y}.parquet"
                for y in range(1910, 1929)])
```

Download everything to disk instead (~9.4 GB):

```bash
hf download omarkhursheed/hansard-gendered-corpus --repo-type dataset \
    --local-dir data --include "corpus/*"
```

## Key statistics

| Metric | Value |
|---|---|
| Total debates | 1,197,828 |
| Total speeches | 6,783,015 |
| Unique speakers (canonical names) | 52,661 |
| Commons speeches | 5,575,783 |
| Commons gender-matched | 4,980,711 (89.3%) |
| Female MPs with speeches (Commons) | 254 |
| Male MPs with speeches (Commons) | 12,395 |

House of Lords gender matching reaches only ~1% (mutable peerage titles);
use the Commons subset for gender analyses.

Full schema, provenance, and known limitations: see DATASHEET.md in the
companion GitHub repository.

## Licensing

Hansard speech text is Crown/Parliamentary material under the Open
Parliament Licence v3.0. Derived metadata is CC BY 4.0.

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
