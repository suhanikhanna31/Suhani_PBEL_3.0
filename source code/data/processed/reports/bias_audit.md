# Bias / fairness audit (writing-style proxies)

**Flagged for closer review:** lexical_diversity, readability_flesch (flagged-rate ratio >= 1.5x between highest and lowest quartile).

## avg_word_length
| Quartile | n messages | mean drift score | flagged rate |
|---|---|---|---|
| Q1 | 502 | 0.044 | 0.026 |
| Q2 | 517 | 0.042 | 0.025 |
| Q3 | 483 | 0.028 | 0.021 |
| Q4 | 498 | 0.035 | 0.022 |

## lexical_diversity
| Quartile | n messages | mean drift score | flagged rate |
|---|---|---|---|
| Q1 | 500 | 0.057 | 0.034 |
| Q2 | 505 | 0.024 | 0.016 |
| Q3 | 995 | 0.034 | 0.022 |

## readability_flesch
| Quartile | n messages | mean drift score | flagged rate |
|---|---|---|---|
| Q1 | 500 | 0.040 | 0.032 |
| Q2 | 502 | 0.020 | 0.010 |
| Q3 | 500 | 0.050 | 0.032 |
| Q4 | 498 | 0.039 | 0.020 |
