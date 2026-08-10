### Eval comparison - `eval500-20260810`

| model | mode | n | schema-valid | found | item F1 | P | R | mean items | price cov | false-find | false give-up | cache hit-rate | policy | eps/min | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grpo-text | paired | 498 | 99.8% | 79.1% | 0.539 | 0.819 | 0.558 | -6.6 | -- | 14 | 90/410 | 93.3% | live | 59.4 | 0 |
| sft-text | paired | 498 | 99.8% | 79.1% | 0.559 | 0.832 | 0.576 | -5.8 | -- | 12 | 92/410 | 94.1% | live | 56.3 | 0 |

_Slice: **all**. `found` means found=true rate for self-report rows and found-flag ACCURACY vs the reference for paired rows. Item F1/P/R and the abstention columns exist only for paired rows -- the teacher is the reference, so it is self-reported by construction. `cache hit-rate` is lookups served from `cache.sqlite`; a low value means the model explored off the warmed distribution._

