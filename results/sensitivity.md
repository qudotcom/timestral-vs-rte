**Exploratory, not pre-registered.** Same test days and models, but the cutoff is moved 12 hours earlier to D-1 00:00 UTC (forecast lead 24 to 47 hours). RTE's dataset does not record when its J-1 forecast is issued, so this checks that the result does not depend on Timestral seeing more recent data than RTE.

Run `20261120T222730Z-sens-1d6743`.

| Model | MAE (MW) | MAPE | MASE vs weekly naive | 10–90% interval coverage | Days better than RTE J-1 |
|---|---:|---:|---:|---:|---:|
| Timestral-28M + calendar covariates | 1,087 | 2.34% | 0.50 | 85% | 65% |
| Timestral-28M (zero-shot) | 1,108 | 2.39% | 0.51 | 85% | 64% |
| RTE official day-ahead (J-1) | 1,216 | 2.66% | 0.56 | — | — |
| Naive: same hour last week | 2,187 | 4.74% | 1.00 | — | 23% |
| Naive: same hour 2 days before | 3,586 | 8.08% | 1.64 | — | 16% |
