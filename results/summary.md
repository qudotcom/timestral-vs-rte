Run `20261120T222645Z-179e47` on duckdb, 2026-11-20T22:26:45+00:00. Protocol SHA-256 `1ad75face9e9959aebe593e65efea465a2bb5b6b6fabd6b8aa1b8293dd9a186b`.
All models scored on the same 2206 hours.

| Model | MAE (MW) | MAPE | MASE vs weekly naive | 10–90% interval coverage | Days better than RTE J-1 |
|---|---:|---:|---:|---:|---:|
| Timestral-28M + calendar covariates | 808 | 1.74% | 0.37 | 85% | 76% |
| Timestral-28M (zero-shot) | 829 | 1.79% | 0.38 | 86% | 78% |
| RTE official day-ahead (J-1) | 1,216 | 2.66% | 0.56 | — | — |
| Naive: same hour last week | 2,187 | 4.74% | 1.00 | — | 23% |
| Naive: same hour 2 days before | 3,586 | 8.08% | 1.64 | — | 16% |
