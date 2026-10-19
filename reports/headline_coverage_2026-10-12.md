# Headline coverage audit — 12 October 2026

**The 16-headline cap substantially restricts the news input. A single 16-versus-64 comparison is worth testing; this audit does not establish that more news will improve forecasts.**

Counted nonblank lines in the downloaded NIFTY `news` field, trimming whitespace and removing exact duplicates within each date, exactly as the current selector does. Primary counts use the actual retained experiment dates; four boundary-excluded source dates are recorded separately. “Available” here means present in the source field, not verified publication availability at our forecast cutoff.

| Split | Retained dates | Median unique headlines/date | All headline occurrences retained at cap 16 | At cap 64 |
|---|---:|---:|---:|---:|
| Training | 1,475 | 78 | 6.0% | 22.4% |
| Validation | 315 | 54 | 30.1% | 97.7% |
| Historical test, descriptive only | 317 | 26 | 60.7% | 100% |

These percentages weight high-volume days more heavily. Training dates range from 11 to 1,551 unique lines; the **average per-date** retained fraction is **20.1% at 16**, **39.2% at 32**, **69.0% at 64**, and **81.1% at 128**. On a median-sized training date, 16 retains about 21% and 64 about 82%. Thus the 6% aggregate figure does not mean that a typical date loses 94% of its news.

**Recommendation: compare 16 with 64 next.** It creates a substantial coverage change on 1,471/1,475 training dates and all validation dates. The hash-based selection is nested, so the original 16 headlines remain included. This is a practical candidate chosen from training coverage and validation feasibility, not a proven optimal cap. Cap 128 adds less per-date coverage and quadruples attention-matrix area relative to 64.

For training/validation, cap 64 requires **72,223 additional unique headlines**, each needing both BGE vectors and TinyBERT sentiment. The existing cache can supply 27,271 of the 99,494 required IDs. Models are already downloaded; no new dataset download is needed. A separate versioned feature cache and configurable headline dimension will be needed before any fits. The attention matrix grows 16-fold relative to cap 16; whole-run time need not grow by that factor.

Coverage shifts substantially across years and splits. More headlines may add relevant information, noise or distribution differences; counts cannot distinguish them. No outcome analysis, feature generation or training was performed. The reviewer verified all 2,111 raw-date counts, 2,107 retained selections, nested caps and cache workloads; 377 protected files remain unchanged.

[Protocol and reproducible audit](../artifacts/diagnostics/headline_coverage_2026-10-12/PROTOCOL.md), [per-date counts](../artifacts/diagnostics/headline_coverage_2026-10-12/daily_counts.csv), [full split/year summary](../artifacts/diagnostics/headline_coverage_2026-10-12/summary.json), [independent review](../artifacts/diagnostics/headline_coverage_2026-10-12/REVIEW.md).
