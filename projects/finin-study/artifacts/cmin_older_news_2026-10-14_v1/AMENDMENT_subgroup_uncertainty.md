# Subgroup uncertainty amendment — 2026-10-14

Decided during new fitting, before inspecting any subgroup contrast. The primary comparison, all data, model training, checkpoints, selection, fit/time budget and full-calendar primary bootstrap remain unchanged.

The original wrapper passes each coverage group's observed target dates to the existing bootstrap. For sparse groups, five observed dates can span more than five trading dates. Retain that original analysis unchanged as a descriptive sensitivity, but use an additional full-period-calendar bootstrap for subgroup intervals reported in the concise report.

For each of the two periods, use the complete matched evaluation target-date grid. Compute group loss-difference sums and case counts on every date; dates without group cases contribute zero sum and zero count. Resample paired circular five-trading-date blocks, keeping companies together. Each draw is total loss-difference sum divided by total sampled group cases. Record zero-case draws rather than treating them as zero loss effects. Average seed losses, not probabilities. Use 2,000 draws with the same fixed RNG seed as the original method. Confirm the all-case interval matches the original primary interval and that subgroup point estimates are unchanged.

This cheap saved-prediction calculation requires no fitting or new inference and does not create a new hypothesis or favorable cohort. Group estimates remain descriptive, correlated and subject to differing composition and weak recent-only coverage (33 cases). Independent final review verifies the supplement and original evidence.
