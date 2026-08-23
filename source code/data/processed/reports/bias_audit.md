# Bias / fairness audit (writing-style proxies)

No proxy showed a >=1.5x disparity in flagged rate across quartiles on this run.

## avg_word_length
| Quartile | n messages | mean drift score | flagged rate |
|---|---|---|---|
| Q1 | 1065 | 0.004 | 0.001 |
| Q2 | 1050 | 0.005 | 0.001 |
| Q3 | 1017 | 0.006 | 0.000 |
| Q4 | 1043 | 0.004 | 0.001 |

## lexical_diversity
| Quartile | n messages | mean drift score | flagged rate |
|---|---|---|---|
| Q1 | 1066 | 0.007 | 0.002 |
| Q2 | 1066 | 0.002 | 0.001 |
| Q3 | 2043 | 0.004 | 0.000 |

## readability_flesch
| Quartile | n messages | mean drift score | flagged rate |
|---|---|---|---|
| Q1 | 1045 | 0.003 | 0.000 |
| Q2 | 1045 | 0.006 | 0.003 |
| Q3 | 1041 | 0.004 | 0.000 |
| Q4 | 1044 | 0.005 | 0.000 |
