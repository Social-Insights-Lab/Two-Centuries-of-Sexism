# Speaker-level aggregation (rebuttal check for Reviewer xe7f)

Source: `speech_classifications.parquet`. Baseline per-speech numbers verified against the paper before aggregating (see JSON for the check list).
Speaker unit = normalized (name, gender); 0 relevant speeches without a parseable speaker excluded from speaker-level rows.

## Speaker concentration

- n_relevant_speeches: 2942
- n_speeches_without_speaker: 0
- n_unique_speakers: 1429
- max_speeches_one_speaker: 32
- median_speeches_per_speaker: 1.0
- top10_speakers_share_pct: 7.7
- speakers_with_one_speech: 926

## Support (stance = for) by gender: per-speech vs one-vote-per-speaker
| gender | n_speeches | n_speakers | for_pct_per_speech | for_pct_speaker_mean |
|---|---|---|---|---|
| F | 389 | 168 | 92.8 | 92.5 |
| M | 2303 | 1158 | 70.1 | 70.3 |

## Speaker-majority stance by gender (chi-square at speaker level)
- contingency: {'F': {'for': 153, 'not_for': 8}, 'M': {'for': 802, 'not_for': 321}}
- chi2 = 39.98, dof = 1, p = 2.57e-10 (43 tied speakers excluded)
| gender | n_speakers | majority_for_pct |
|---|---|---|
| F | 161 | 95.0 |
| M | 1123 | 71.4 |

## Speaker-majority sexism by stance (see script comment on interpretation)
| stance | n_speakers | majority_sexist_pct | exact_tie_pct | ever_sexist_pct |
|---|---|---|---|---|
| for | 1104 | 18.5 | 5.3 | 32.1 |
| against | 344 | 47.4 | 5.2 | 55.5 |
| both | 159 | 56.0 | 6.9 | 64.2 |

## Support by era and gender
| era | F_per_speech | F_speaker_mean | F_n_speakers | M_per_speech | M_speaker_mean | M_n_speakers |
|---|---|---|---|---|---|---|
| pre-1870 | None | None | 0 | 57.1 | 54.8 | 31 |
| 1870-1899 | None | None | 0 | 54.4 | 53.0 | 224 |
| 1900-1918 | None | None | 0 | 64.0 | 63.0 | 382 |
| 1919-1928 | 91.7 | 95.2 | 6 | 73.9 | 73.2 | 209 |
| 1929-1950 | 96.6 | 91.7 | 16 | 82.4 | 81.6 | 125 |
| post-1950 | 92.2 | 92.7 | 151 | 90.6 | 89.0 | 276 |

## Sexism rates by stance
| stance | n_speeches | any_per_speech | any_speaker_mean | n_speakers | hostile_per_speech | hostile_speaker_mean | benevolent_per_speech | benevolent_speaker_mean |
|---|---|---|---|---|---|---|---|---|
| for | 2167 | 21.3 | 22.7 | 1104 | 4.1 | 4.3 | 19.0 | 20.4 |
| against | 570 | 54.0 | 48.9 | 344 | 43.5 | 38.6 | 34.0 | 29.4 |
| both | 205 | 56.6 | 59.4 | 159 | 27.3 | 29.3 | 48.8 | 51.7 |

## Mix within sexist speeches
| stance | n_sexist_speeches | hostile_only_per_speech | hostile_only_speaker_mean | n_speakers | benevolent_only_per_speech | benevolent_only_speaker_mean | both_hb_per_speech | both_hb_speaker_mean |
|---|---|---|---|---|---|---|---|---|
| for | 462 | 10.8 | 10.2 | 354 | 81.0 | 81.4 | 8.2 | 8.3 |
| against | 308 | 37.0 | 39.1 | 191 | 19.5 | 22.2 | 43.5 | 38.7 |
| both | 116 | 13.8 | 13.9 | 102 | 51.7 | 51.1 | 34.5 | 35.0 |

## Hostile share of sexist speeches by era
| era | n_sexist_speeches | n_speakers | hostile_share_per_speech | hostile_share_speaker_mean |
|---|---|---|---|---|
| pre-1870 | 12 | 11 | 41.7 | 40.9 |
| 1870-1899 | 199 | 130 | 59.8 | 55.4 |
| 1900-1918 | 315 | 199 | 48.6 | 43.1 |
| 1919-1928 | 121 | 93 | 38.8 | 34.2 |
| 1929-1950 | 101 | 72 | 26.7 | 28.7 |
| post-1950 | 138 | 115 | 29.7 | 30.9 |
