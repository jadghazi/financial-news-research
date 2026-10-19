"""Write the concise older-news finding from completed saved evidence, no fitting."""
import json
from pathlib import Path


def main():
    root = Path('artifacts/cmin_older_news_2026-10-14_v1')
    s = json.loads((root / 'analysis/summary.json').read_text())
    subgroup_supplement = json.loads((root / 'subgroup_calendar/summary.json').read_text())
    calendar_groups = {g['group']: g for g in subgroup_supplement['groups']}
    path = Path('reports/cmin_older_news_2026-10-14.md')
    if path.exists():
        raise ValueError('Existing report preserved; refusing overwrite')
    p = s['primary']
    comps = {(v['first'], v['second']): v for v in s['comparisons']}
    def effect_text(first, second):
        c = comps[(first, second)]
        return f"{c['delta_log_loss']:+.6f} (interval [{c['conditional_95_interval'][0]:+.6f}, {c['conditional_95_interval'][1]:+.6f}])"
    useful_overall = s['candidate_useful_relative_addition'] and all(
        comps[('all_news', reference)]['delta_log_loss'] < 0
        and comps[('all_news', reference)]['conditional_95_interval'][1] < 0
        for reference in ('price_only', 'training_prior'))
    if useful_overall:
        verdict = 'Older-news addition showed a candidate useful forecasting benefit in this exploratory study; timing/content controls and untouched confirmation remain required.'
    elif s['candidate_useful_relative_addition']:
        verdict = 'Older news helped relative to current-bucket news under the declared criterion, but did not establish an overall useful forecaster against price and prevalence references.'
    else:
        verdict = 'No robust, practically useful older-news addition was established in this fixed predictive setup.'
    lines = ['# Does older available news add forecasting value? — 2026-10-14', '', '**' + verdict + '**', '',
        'We compared three otherwise matched 96-input pooled models: prices plus fixed company identity; the same prices plus current-arrival-bucket news; and the same prices plus current and four older arrival buckets. All retain five price sessions. Both news models use the same FININ-style headline/sentiment fusion, self-attention, market-query weighting and uniform temporal pooling. The no-news block is zero; its news parameters are unused, so total parameter equality does not imply equal active capacity.', '',
        f"Only eight new fits were run, taking {s['execution']['total_new_neural_seconds']:.1f} seconds under the 1,200-second ceiling. All four matching all-news fits from the preceding decay study were reused, without choosing favorable seeds or retraining. Identical initial states, hyperparameters, purged plans and training-only transforms were checked. No new acquisition, encoding or convex fitting; no tuning or automatic retries.", '',
        '## Matched forecast results', '',
        f"{s['evaluation_company_days']:,} company-days on {s['evaluation_unique_dates']} dates; two chronological periods and seeds 17/42. Lower log loss is better. Values average raw seed-model losses, not ensemble probabilities.", '',
        '| Inputs | Log loss | Accuracy |', '|---|---:|---:|']
    for name, label in [('price_only', 'Prices (matched pooled geometry)'), ('recent_news', 'Prices + current arrival bucket'),
                        ('all_news', 'Prices + current and four older buckets'), ('training_prior', 'Constant training-prevalence probabilities')]:
        v = s['metrics'][name]
        lines.append(f"| {label} | {v['mean_seed_log_loss']:.6f} | {v['mean_seed_accuracy']:.2%} |")
    lines += ['', f"**Adding older news: log-loss difference {p['delta_log_loss']:+.6f}, conditional 95% interval [{p['conditional_95_interval'][0]:+.6f}, {p['conditional_95_interval'][1]:+.6f}].** The predeclared practical decrease was 0.005. A useful relative addition also required improvement in both period means and at least three of four seed-period pairs, with the interval below zero.", '',
              '| Evaluation period | Seed | All-news minus current-news loss |', '|---|---:|---:|']
    for v in p['seed_period_differences']:
        period = 'May–August 2020' if v['fold'] == 0 else 'January–April 2021'
        lines.append(f"| {period} | {v['seed']} | {v['delta_log_loss']:+.6f} |")
    lines += ['', 'The interval uses 2,000 circular five-date blocks within periods, keeping stocks together. It conditions on saved fits and omits training uncertainty. Raw neural and validation-selected constant-fallback systems are retained separately in the analysis.', '',
              '## Coverage checks declared before fitting', '',
              '| Available-news group | Company-days | Dates | All-news minus current-news loss | Conditional 95% interval |',
              '|---|---:|---:|---:|---|']
    labels = {'recent_and_older': 'Both current and older', 'older_only': 'Older only', 'recent_only': 'Current only', 'empty': 'No available news'}
    for g in s['groups']:
        c = calendar_groups[g['group']]
        delta = f"{c['delta_log_loss']:+.6f}" if c else 'NA'
        ci = f"[{c['conditional_95_interval'][0]:+.6f}, {c['conditional_95_interval'][1]:+.6f}]" if c else 'NA'
        lines.append(f"| {labels[g['group']]} | {g['company_days']} | {g['unique_dates']} | {delta} | {ci} |")
    lines += ['', 'These groups use masks alone, not outcomes, and are descriptive. Their reported intervals use the full period trading-date calendars, with zero sums/counts on dates without group cases, as specified in a pre-group-results uncertainty amendment. Original five-observed-group-date intervals remain preserved as a descriptive sensitivity. Adding older buckets changes coverage, sentiment, pooling shares and fitted parameters. Group composition differs; comparisons do not isolate older headline semantics or a causal news effect. The current-only group has just 33 cases and is particularly uncertain.', '',
              'Even on empty-news cases, separately trained models can predict differently because they learned different shared parameters; this does not imply hidden news inputs on those rows.', '',
              '## Class behavior', '', '| Period | Inputs | Seed | UP / cases | Selected epoch | Constant fallback |', '|---|---|---:|---:|---:|---|']
    for f in s['fits']:
        period = '2020' if f['fold'] == 0 else '2021'
        lines.append(f"| {period} | {f['arm']} | {f['seed']} | {f['raw']['predicted_up']} / {f['raw']['count']} | {f['selected_epoch']} | {f['constant_fallback']} |")
    lines += ['', '## Interpretation and stopping decision', '', verdict, '',
        'An improvement over a weak current-news model alone is insufficient to establish useful forecasting. All-news versus the newly matched price-only model and constant probabilities are secondary references; complete contrasts, Brier/AUC/MCC, probability spread and selected-system results are retained in the analysis. No favorable subgroup, additional seed or model was pursued.', '',
        f"The all-news model also had higher loss than matched price-only by {effect_text('all_news', 'price_only')}, a small secondary difference below the practical threshold. Current-bucket news improved on prices by only {effect_text('recent_news', 'price_only')} and on constant probabilities by {effect_text('recent_news', 'training_prior')}. Those intervals include zero and both contrasts reverse sign between periods; its lowest point-estimate loss does not establish a genuine forecasting gain.", '',
        '"Current/newest" means the cutoff session’s proxy-arrival bucket, including empty buckets, not the latest nonempty bucket or verified newly published news. "Older" means 1–4 trading sessions since proxy arrival. Source-date D+2 availability/timezones and publication dates remain unverified. Development periods were previously inspected, so the entire comparison is exploratory; reserved late-2021 outcomes remain unopened.', '',
        '[FININ](https://aclanthology.org/2024.findings-emnlp.189.pdf), sections 4.2–4.3 and Eq. 7, uses historical daily news representations. This experiment tests an independent pooled public-data adaptation and a narrower predictive question. It does not estimate publication-time market response, demonstrate information diffusion or identify an economic half-life.', '',
        f"Evidence: [frozen protocol](../docs/cmin_older_news_protocol_2026-10-14.md), [configuration](../configs/cmin_older_news_v1.json), `{root.as_posix()}/analysis/summary.json`, paired per-case logits and original/new checkpoints. All {s['baseline_preservation']['unchanged_files']:,} protected prior-study files were verified unchanged. Independent review status: " + ('complete; see FINAL_REVIEW.md. Read-only conduct was not sandbox-enforced.' if (root/'FINAL_REVIEW.md').exists() else 'pending; completed review will be retained with the study.')]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(path), 'candidate_useful_relative_addition': s['candidate_useful_relative_addition'],
                      'useful_overall_forecaster_established': bool(useful_overall), 'verdict': verdict}, indent=2))


if __name__ == '__main__':
    main()
