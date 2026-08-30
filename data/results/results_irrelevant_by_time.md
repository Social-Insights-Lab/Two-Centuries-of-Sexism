# Irrelevant-rate distribution (rebuttal check for Reviewer 9ZNL)

Source: `speech_classifications.parquet` (n=6,531; 3,589 irrelevant = 55.0%).
Tier re-derived from target_text with the canonical extractor logic; 0 speeches matched neither tier.

## By decade
| decade | n | n_irrelevant | irrelevant_pct |
|---|---|---|---|
| 1800.0 | 1.0 | 1.0 | 100.0 |
| 1810.0 | 10.0 | 10.0 | 100.0 |
| 1820.0 | 8.0 | 8.0 | 100.0 |
| 1830.0 | 37.0 | 31.0 | 83.8 |
| 1840.0 | 53.0 | 48.0 | 90.6 |
| 1850.0 | 54.0 | 44.0 | 81.5 |
| 1860.0 | 89.0 | 71.0 | 79.8 |
| 1870.0 | 243.0 | 82.0 | 33.7 |
| 1880.0 | 227.0 | 121.0 | 53.3 |
| 1890.0 | 305.0 | 120.0 | 39.3 |
| 1900.0 | 409.0 | 199.0 | 48.7 |
| 1910.0 | 1396.0 | 556.0 | 39.8 |
| 1920.0 | 576.0 | 283.0 | 49.1 |
| 1930.0 | 563.0 | 344.0 | 61.1 |
| 1940.0 | 348.0 | 249.0 | 71.6 |
| 1950.0 | 257.0 | 205.0 | 79.8 |
| 1960.0 | 298.0 | 236.0 | 79.2 |
| 1970.0 | 313.0 | 224.0 | 71.6 |
| 1980.0 | 413.0 | 288.0 | 69.7 |
| 1990.0 | 671.0 | 374.0 | 55.7 |
| 2000.0 | 260.0 | 95.0 | 36.5 |

## By era
| era | n | n_irrelevant | irrelevant_pct |
|---|---|---|---|
| pre-1870 | 252 | 213 | 84.5 |
| 1870-1899 | 775 | 323 | 41.7 |
| 1900-1918 | 1680 | 716 | 42.6 |
| 1919-1928 | 676 | 302 | 44.7 |
| 1929-1950 | 957 | 631 | 65.9 |
| post-1950 | 2191 | 1404 | 64.1 |

## By keyword tier
| tier | n | n_irrelevant | irrelevant_pct |
|---|---|---|---|
| tier1 | 2725 | 1307 | 48.0 |
| tier2 | 3806 | 2282 | 60.0 |

## Tier x era
| era | tier | n | n_irrelevant | irrelevant_pct |
|---|---|---|---|---|
| pre-1870 | tier1 | 150 | 124 | 82.7 |
| pre-1870 | tier2 | 102 | 89 | 87.3 |
| 1870-1899 | tier1 | 410 | 154 | 37.6 |
| 1870-1899 | tier2 | 365 | 169 | 46.3 |
| 1900-1918 | tier1 | 1103 | 418 | 37.9 |
| 1900-1918 | tier2 | 577 | 298 | 51.6 |
| 1919-1928 | tier1 | 252 | 71 | 28.2 |
| 1919-1928 | tier2 | 424 | 231 | 54.5 |
| 1929-1950 | tier1 | 252 | 160 | 63.5 |
| 1929-1950 | tier2 | 705 | 471 | 66.8 |
| post-1950 | tier1 | 558 | 380 | 68.1 |
| post-1950 | tier2 | 1633 | 1024 | 62.7 |

## By chamber
| chamber | n | n_irrelevant | irrelevant_pct |
|---|---|---|---|
| Commons | 6531 | 3589 | 55.0 |

## By speaker gender
| gender | n | n_irrelevant | irrelevant_pct |
|---|---|---|---|
| F | 611 | 222 | 36.3 |
| M | 5430 | 3127 | 57.6 |
| nan | 490 | 240 | 49.0 |

## Era composition of corpus, before vs after the relevance filter
| era | share_pre_filter_pct | share_post_filter_pct | shift_pp |
|---|---|---|---|
| pre-1870 | 3.9 | 1.3 | -2.6 |
| 1870-1899 | 11.9 | 15.4 | 3.5 |
| 1900-1918 | 25.7 | 32.8 | 7.1 |
| 1919-1928 | 10.4 | 12.7 | 2.3 |
| 1929-1950 | 14.7 | 11.1 | -3.6 |
| post-1950 | 33.5 | 26.8 | -6.7 |

## Chi-square tests (grouping variable vs irrelevant flag)

- decade: chi2 = 599.64, dof = 20, p = 3.42e-114, min expected cell = 0.5
- era: chi2 = 396.65, dof = 5, p = 1.57e-83, min expected cell = 113.5
- tier: chi2 = 91.81, dof = 1, p = 9.54e-22, min expected cell = 1227.5

## Median word count
- relevant: 658
- irrelevant: 996
