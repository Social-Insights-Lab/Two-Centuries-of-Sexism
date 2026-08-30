# Datasheet: Hansard Gender-Matched Corpus

An organized, speaker-enriched version of the Historic Hansard corpus of UK
parliamentary debates (1803-2005), built for computational social science
research. Speech and debate records are partitioned by year as parquet files.

All statistics below were recomputed directly from the released files.

## Summary statistics

| Metric | Value |
|---|---|
| Total debates | 1,197,828 |
| Total speeches | 6,783,015 |
| Unique speakers (canonical names) | 52,661 |
| Date range | 1803-2005 |
| Chambers | Commons, Lords |

House of Commons subset:

| Metric | Value |
|---|---|
| Speeches | 5,575,783 |
| Gender-matched speeches | 4,980,711 (89.3%) |
| Male MPs with speeches | 12,395 |
| Female MPs with speeches | 254 |
| Male speeches | 4,829,844 |
| Female speeches | 150,867 |

House of Lords gender matching reaches only ~1% because peerage titles are
mutable (one individual may appear under several titles); analyses in the
paper are restricted to the Commons.

## Files

```
corpus/
  speeches/speeches_YYYY.parquet   # 201 files, one per year
  debates/debates_YYYY.parquet     # 201 files, one per year
  house_members_gendered.parquet   # unified MP database with gender
womens_rights/
  corpus_with_context.parquet       # 6,531 speeches + context windows
  speech_classifications.parquet    # LLM stance and sexism labels
```

## Speeches schema

| Field | Type | Description |
|---|---|---|
| speech_id | str | Unique speech identifier |
| debate_id | str | Parent debate identifier |
| file_path | str | Source HTML path in Historic Hansard |
| sequence_number | int | Position within debate |
| speaker | str | Speaker name as printed in Hansard |
| normalized_speaker | str | Lowercased name used for matching |
| canonical_name | str | Clean display name (from MP database if matched) |
| person_id | str | MP identifier (uk.org.publicwhip format), if matched |
| gender | str | M / F for matched MPs, null otherwise |
| matched_mp | bool | Whether speaker matched to the MP database |
| party | str | Party of matched MP |
| constituency | str | Constituency of matched MP |
| contribution_id | str | Hansard contribution identifier |
| is_question | bool | Whether the speech is a parliamentary question |
| question_number | str | Question number, if applicable |
| text | str | Full speech text |
| word_count | int | Word count of text |
| year / decade / month / date | int, str | Date of debate |
| chamber | str | Commons or Lords |
| title / topic | str | Debate title and topic |
| hansard_reference / reference_volume / reference_columns | str | Official Hansard citation |

## Debates schema

One row per debate with title, topic, Hansard reference, full text, word and
speech counts, and speaker composition: parallel lists of speaker names,
normalized names, canonical names, person ids, and genders, plus aggregate
counts (confirmed_mps, male_mps, female_mps, gender_ratio, has_female,
has_male).

## Women's rights subset schema

`corpus_with_context.parquet` (6,531 rows): speech_id, debate_id, target_text,
preceding_speeches, following_speeches, context_text (up to 5 speeches either
side of the target).

`speech_classifications.parquet` (6,531 rows): speech metadata (speaker,
canonical_name, year, decade, era, date, gender, party, chamber, word_count,
target_text, context_text) plus model outputs:

| Field | Description |
|---|---|
| llm_stance | for / against / both / irrelevant |
| llm_stance_confidence | Model-reported confidence |
| llm_stance_rationale | Free-text rationale |
| llm_hostile | Hostile sexism flag (null for 14 speeches where the model returned schema-incomplete output; treat null as False to match the paper) |
| llm_hostile_subcategories | dominative_paternalism, competitive_gender_differentiation, heterosexual_hostility |
| llm_hostile_quote | Verbatim supporting quote |
| llm_benevolent | Benevolent sexism flag (same null caveat) |
| llm_benevolent_subcategories | protective_paternalism, complementary_gender_differentiation, heterosexual_intimacy |
| llm_benevolent_quote | Verbatim supporting quote |

Labels were produced by Claude Sonnet 4.6 with a two-pass prompt (stance
first; sexism only for non-irrelevant speeches), validated against a
300-speech expert-annotated set (stance kappa 0.711 vs human consensus;
see the paper for full validation detail including sexism agreement).

Subset speech_ids join the corpus/ files directly for 6,501 of 6,531
speeches; the remaining 30 have slightly shifted speech numbering within the
same debate_id (speech-segmentation differences between corpus builds) and
can be located by matching target_text within the debate.

## Provenance

1. Debate HTML crawled from Historic Hansard (public government records).
2. Speeches parsed and split per speaker turn.
3. MP database aggregated from EveryPolitician, MySociety, WikiData, and
   Wikipedia MP lists; gender from explicit records, honorifics, WikiData,
   the pre-1918 all-male rule, and the Gender Guesser library (97% accuracy
   on a labelled subset).
4. Speakers matched to MPs by cascaded exact, title-constrained, temporal,
   and fuzzy matching, tuned for precision over recall to avoid misgendering.
5. Suffrage-related speeches extracted by two-tier keyword search
   (Appendix A of the paper), then classified by LLM.

## Known limitations

- Lords speeches are mostly unmatched (~1%); use the Commons for gender
  analyses.
- Match rates are lower before 1850 (limited MP records, procedural
  speakers).
- Common surnames can be ambiguous; matching prefers precision but errors
  remain. A noise-induction check in the paper shows headline results are
  robust to 3% gender-label noise.
- Speaker gender is binary (M/F), reflecting what historical records
  support, not the full range of gender identities.
- OCR and transcription artifacts from the original Hansard digitization
  are present in speech text.

## Licensing

Hansard text: Open Parliament Licence v3.0. Annotations, labels, and derived
metadata: CC BY 4.0. Code: MIT.
