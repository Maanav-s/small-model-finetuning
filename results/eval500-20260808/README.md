### Eval comparison - `eval500-20260808`

| model | mode | n | schema-valid | found | item F1 | P | R | mean items | price cov | false-find | false give-up | cache hit-rate | policy | eps/min | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-235B-A22B-Instruct-2507-FP8 | self-report | 500 | 99.6% | 82.0% | -- | -- | -- | 34.5 | 0.722 | -- | -- | -- | live | -- | -- |
| sft-text | paired | 498 | 100.0% | 81.3% | 0.560 | 0.827 | 0.573 | -7.8 | -- | 10 | 83/410 | 94.0% | live | 40.1 | 0 |
| base-text | paired | 498 | 95.4% | 74.1% | 0.438 | 0.737 | 0.459 | -10.4 | -- | 29 | 100/410 | 92.1% | live | 16.4 | 0 |

_Slice: **all**. `found` means found=true rate for self-report rows and found-flag ACCURACY vs the reference for paired rows. Item F1/P/R and the abstention columns exist only for paired rows -- the teacher is the reference, so it is self-reported by construction. `cache hit-rate` is lookups served from `cache.sqlite`; a low value means the model explored off the warmed distribution._

