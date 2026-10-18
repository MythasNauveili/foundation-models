# Saved forecasting comparisons

Validated schema-v5 results. Each track uses its full schedule. Lower scores are better; improvement compares aggregate primary scores with the fixed reference. Zero-shot does not establish absence from pretraining. CPU timings exclude loading/downloads.

## Strasbourg weather · daily · 7 steps (weather-daily-7-v1)

Split **exploratory** · reference **Seasonal naïve** · main score **MAE**.

### Target only

| Model ID | Model | MAE | MASE | Improvement (%) | WIS | Coverage80 (%) | Windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| timestral-209m | Timestral-209M | 2.637 | 0.552 | 40.530 | 1.695 | 85.714 | 12 |
| timestral-123m | Timestral-123M | 2.644 | 0.553 | 40.371 | 1.690 | 86.905 | 12 |
| timestral-28m | Timestral-28M | 2.742 | 0.574 | 38.157 | 1.751 | 85.714 | 12 |
| timestral-5m | Timestral-5M | 2.760 | 0.577 | 37.753 | 1.774 | 85.714 | 12 |
| chronos2 | Chronos-2 | 2.862 | 0.600 | 35.470 | 1.812 | 79.762 | 12 |
| baseline:Persistence | Persistence | 3.248 | 0.676 | 26.765 | — | — | 12 |
| baseline:Seasonal naïve | Seasonal naïve | 4.435 | 0.934 | 0.000 | — | — | 12 |

### Calendar

| Model ID | Model | MAE | MASE | Improvement (%) | WIS | Coverage80 (%) | Windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| timestral-209m | Timestral-209M | 2.607 | 0.545 | 41.207 | 1.678 | 89.286 | 12 |
| timestral-123m | Timestral-123M | 2.632 | 0.551 | 40.653 | 1.691 | 88.095 | 12 |
| timestral-28m | Timestral-28M | 2.643 | 0.553 | 40.392 | 1.706 | 86.905 | 12 |
| chronos2 | Chronos-2 | 2.745 | 0.575 | 38.096 | 1.735 | 79.762 | 12 |
| timestral-5m | Timestral-5M | 2.896 | 0.605 | 34.697 | 1.868 | 85.714 | 12 |
| baseline:Persistence | Persistence | 3.248 | 0.676 | 26.765 | — | — | 12 |
| baseline:Seasonal naïve | Seasonal naïve | 4.435 | 0.934 | 0.000 | — | — | 12 |

### Past observed weather

| Model ID | Model | MAE | MASE | Improvement (%) | WIS | Coverage80 (%) | Windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| timestral-123m | Timestral-123M | 2.634 | 0.551 | 40.600 | 1.685 | 86.905 | 12 |
| timestral-209m | Timestral-209M | 2.672 | 0.559 | 39.745 | 1.690 | 88.095 | 12 |
| timestral-28m | Timestral-28M | 2.716 | 0.569 | 38.746 | 1.741 | 85.714 | 12 |
| timestral-5m | Timestral-5M | 2.779 | 0.581 | 37.329 | 1.785 | 85.714 | 12 |
| chronos2 | Chronos-2 | 2.836 | 0.594 | 36.057 | 1.799 | 80.952 | 12 |
| baseline:Persistence | Persistence | 3.248 | 0.676 | 26.765 | — | — | 12 |
| baseline:Seasonal naïve | Seasonal naïve | 4.435 | 0.934 | 0.000 | — | — | 12 |

### Past observed weather + calendar

| Model ID | Model | MAE | MASE | Improvement (%) | WIS | Coverage80 (%) | Windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| timestral-123m | Timestral-123M | 2.618 | 0.548 | 40.970 | 1.687 | 89.286 | 12 |
| timestral-209m | Timestral-209M | 2.624 | 0.549 | 40.826 | 1.686 | 90.476 | 12 |
| timestral-28m | Timestral-28M | 2.635 | 0.552 | 40.578 | 1.708 | 86.905 | 12 |
| chronos2 | Chronos-2 | 2.813 | 0.589 | 36.573 | 1.786 | 80.952 | 12 |
| timestral-5m | Timestral-5M | 2.945 | 0.615 | 33.590 | 1.893 | 84.524 | 12 |
| baseline:Persistence | Persistence | 3.248 | 0.676 | 26.765 | — | — | 12 |
| baseline:Seasonal naïve | Seasonal naïve | 4.435 | 0.934 | 0.000 | — | — | 12 |

## Bike demand · every two hours · 12 steps (bikes-2hourly-12-v1)

Split **exploratory** · reference **Seasonal naïve** · main score **MAE**.

### Target only

| Model ID | Model | MAE | MASE | Improvement (%) | WIS | Coverage80 (%) | Windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chronos2 | Chronos-2 | 61.949 | 0.613 | 51.965 | 44.346 | 78.472 | 12 |
| timestral-123m | Timestral-123M | 64.819 | 0.636 | 49.739 | 44.919 | 84.722 | 12 |
| timestral-209m | Timestral-209M | 69.624 | 0.683 | 46.013 | 44.811 | 83.333 | 12 |
| timestral-28m | Timestral-28M | 90.331 | 0.900 | 29.957 | 58.643 | 84.028 | 12 |
| baseline:Four-week average | Four-week average | 93.841 | 0.911 | 27.235 | — | — | 12 |
| timestral-5m | Timestral-5M | 101.311 | 1.003 | 21.443 | 63.738 | 84.028 | 12 |
| baseline:Seasonal naïve | Seasonal naïve | 128.965 | 1.252 | 0.000 | — | — | 12 |
| baseline:Persistence | Persistence | 340.472 | 3.360 | -164.003 | — | — | 12 |

### Calendar

| Model ID | Model | MAE | MASE | Improvement (%) | WIS | Coverage80 (%) | Windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| timestral-123m | Timestral-123M | 70.230 | 0.690 | 45.543 | 51.852 | 87.500 | 12 |
| chronos2 | Chronos-2 | 71.934 | 0.713 | 44.222 | 48.916 | 80.556 | 12 |
| timestral-209m | Timestral-209M | 73.324 | 0.721 | 43.145 | 54.785 | 81.944 | 12 |
| timestral-28m | Timestral-28M | 78.236 | 0.777 | 39.335 | 55.199 | 87.500 | 12 |
| baseline:Four-week average | Four-week average | 93.841 | 0.911 | 27.235 | — | — | 12 |
| timestral-5m | Timestral-5M | 102.652 | 1.015 | 20.404 | 65.954 | 88.194 | 12 |
| baseline:Seasonal naïve | Seasonal naïve | 128.965 | 1.252 | 0.000 | — | — | 12 |
| baseline:Persistence | Persistence | 340.472 | 3.360 | -164.003 | — | — | 12 |

### Past observed weather

| Model ID | Model | MAE | MASE | Improvement (%) | WIS | Coverage80 (%) | Windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chronos2 | Chronos-2 | 68.875 | 0.680 | 46.594 | 47.060 | 78.472 | 12 |
| timestral-123m | Timestral-123M | 74.528 | 0.735 | 42.211 | 51.236 | 84.722 | 12 |
| timestral-209m | Timestral-209M | 74.568 | 0.731 | 42.180 | 49.941 | 79.861 | 12 |
| baseline:Four-week average | Four-week average | 93.841 | 0.911 | 27.235 | — | — | 12 |
| timestral-28m | Timestral-28M | 94.453 | 0.938 | 26.761 | 60.797 | 83.333 | 12 |
| timestral-5m | Timestral-5M | 101.998 | 1.010 | 20.911 | 64.531 | 84.722 | 12 |
| baseline:Seasonal naïve | Seasonal naïve | 128.965 | 1.252 | 0.000 | — | — | 12 |
| baseline:Persistence | Persistence | 340.472 | 3.360 | -164.003 | — | — | 12 |

### Past observed weather + calendar

| Model ID | Model | MAE | MASE | Improvement (%) | WIS | Coverage80 (%) | Windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chronos2 | Chronos-2 | 70.161 | 0.696 | 45.597 | 48.377 | 77.778 | 12 |
| timestral-28m | Timestral-28M | 75.992 | 0.754 | 41.076 | 53.833 | 85.417 | 12 |
| timestral-123m | Timestral-123M | 78.989 | 0.779 | 38.752 | 56.004 | 86.806 | 12 |
| timestral-209m | Timestral-209M | 82.683 | 0.811 | 35.887 | 58.032 | 81.944 | 12 |
| baseline:Four-week average | Four-week average | 93.841 | 0.911 | 27.235 | — | — | 12 |
| timestral-5m | Timestral-5M | 101.818 | 1.009 | 21.050 | 64.952 | 88.194 | 12 |
| baseline:Seasonal naïve | Seasonal naïve | 128.965 | 1.252 | 0.000 | — | — | 12 |
| baseline:Persistence | Persistence | 340.472 | 3.360 | -164.003 | — | — | 12 |

## Retail revenue · monthly · 6 steps (retail-monthly-6-v1)

Split **exploratory** · reference **Seasonal naïve** · main score **MAE**.

### Target only

| Model ID | Model | MAE | MASE | Improvement (%) | WIS | Coverage80 (%) | Windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chronos2 | Chronos-2 | 11128.128 | 0.436 | 27.011 | 6625.097 | 62.500 | 4 |
| timestral-123m | Timestral-123M | 12619.018 | 0.495 | 17.232 | 7703.648 | 79.167 | 4 |
| timestral-209m | Timestral-209M | 12642.443 | 0.496 | 17.079 | 7516.656 | 83.333 | 4 |
| timestral-28m | Timestral-28M | 13966.818 | 0.548 | 8.392 | 8779.762 | 66.667 | 4 |
| timestral-5m | Timestral-5M | 15102.456 | 0.593 | 0.943 | 8751.882 | 83.333 | 4 |
| baseline:Seasonal naïve | Seasonal naïve | 15246.292 | 0.595 | 0.000 | — | — | 4 |
| baseline:Persistence | Persistence | 55953.958 | 2.196 | -267.000 | — | — | 4 |

### Calendar

| Model ID | Model | MAE | MASE | Improvement (%) | WIS | Coverage80 (%) | Windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| timestral-209m | Timestral-209M | 11029.875 | 0.433 | 27.655 | 7601.244 | 91.667 | 4 |
| timestral-123m | Timestral-123M | 11774.237 | 0.462 | 22.773 | 7578.146 | 100.000 | 4 |
| timestral-5m | Timestral-5M | 12207.940 | 0.479 | 19.928 | 7847.678 | 100.000 | 4 |
| chronos2 | Chronos-2 | 13061.932 | 0.511 | 14.327 | 8598.937 | 62.500 | 4 |
| timestral-28m | Timestral-28M | 14062.664 | 0.551 | 7.763 | 8616.336 | 91.667 | 4 |
| baseline:Seasonal naïve | Seasonal naïve | 15246.292 | 0.595 | 0.000 | — | — | 4 |
| baseline:Persistence | Persistence | 55953.958 | 2.196 | -267.000 | — | — | 4 |

## RTE regional electricity · hourly · 24 steps (rte-hourly-24-v1)

Split **development** · reference **Weekly naïve** · main score **MASE**.

### Target only

| Model ID | Model | MAE | MASE | Improvement (%) | WIS | Coverage80 (%) | Windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| timestral-209m | Timestral-209M | 136.573 | 0.427 | 60.594 | 89.459 | 80.729 | 120 |
| timestral-123m | Timestral-123M | 141.553 | 0.445 | 58.954 | 92.833 | 78.403 | 120 |
| chronos2 | Chronos-2 | 148.858 | 0.460 | 57.614 | 100.273 | 70.799 | 120 |
| timestral-28m | Timestral-28M | 154.939 | 0.471 | 56.544 | 104.365 | 78.819 | 120 |
| timestral-5m | Timestral-5M | 167.168 | 0.514 | 52.625 | 114.499 | 77.604 | 120 |
| baseline:Weekly naïve | Weekly naïve | 369.213 | 1.085 | 0.000 | — | — | 120 |
| baseline:Four-week average | Four-week average | 395.424 | 1.114 | -2.684 | — | — | 120 |
| baseline:Daily naïve | Daily naïve | 324.124 | 1.266 | -16.689 | — | — | 120 |
| baseline:Persistence | Persistence | 534.331 | 2.264 | -108.729 | — | — | 120 |

### Calendar

| Model ID | Model | MAE | MASE | Improvement (%) | WIS | Coverage80 (%) | Windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| timestral-209m | Timestral-209M | 138.980 | 0.435 | 59.920 | 90.671 | 82.431 | 120 |
| timestral-123m | Timestral-123M | 141.197 | 0.449 | 58.606 | 92.131 | 79.236 | 120 |
| timestral-28m | Timestral-28M | 147.754 | 0.461 | 57.478 | 98.282 | 79.132 | 120 |
| chronos2 | Chronos-2 | 147.459 | 0.466 | 57.073 | 98.777 | 72.569 | 120 |
| timestral-5m | Timestral-5M | 161.649 | 0.501 | 53.775 | 109.732 | 79.444 | 120 |
| baseline:Weekly naïve | Weekly naïve | 369.213 | 1.085 | 0.000 | — | — | 120 |
| baseline:Four-week average | Four-week average | 395.424 | 1.114 | -2.684 | — | — | 120 |
| baseline:Daily naïve | Daily naïve | 324.124 | 1.266 | -16.689 | — | — | 120 |
| baseline:Persistence | Persistence | 534.331 | 2.264 | -108.729 | — | — | 120 |

