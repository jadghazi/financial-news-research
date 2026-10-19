"""Full-calendar subgroup uncertainty supplement from saved logits; no ML runs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.cmin_panel_data import read, write_once
from src.cmin_panel_experiment import sha256_file

ROOT = Path('artifacts/cmin_older_news_2026-10-14_v1')


def calculate(records, group=None, samples=2000):
    rng = np.random.default_rng(20261014)
    grids, raw = [], []
    for record in records:
        y = record['targets']
        a = np.mean(np.logaddexp(0, record['all_news']) - y * record['all_news'], axis=0)
        b = np.mean(np.logaddexp(0, record['recent_news']) - y * record['recent_news'], axis=0)
        diff = a - b
        mask = np.ones(len(y), dtype=bool) if group is None else record['groups'] == group
        raw.extend(diff[mask].tolist())
        dates = record['dates']
        grid = np.asarray([[diff[(dates == date) & mask].sum(), int(((dates == date) & mask).sum())]
                           for date in sorted(set(dates))], dtype=float)
        grids.append(grid)
    estimates, empty = [], 0
    for _ in range(samples):
        total, count = 0., 0.
        for grid in grids:
            starts = rng.integers(0, len(grid), size=(len(grid) + 4) // 5)
            indices = ((starts[:, None] + np.arange(5)) % len(grid)).ravel()[:len(grid)]
            values = grid[indices].sum(0)
            total += values[0]
            count += values[1]
        if count:
            estimates.append(total / count)
        else:
            empty += 1
    return {'group': group or 'all_cases', 'company_days': len(raw),
        'delta_log_loss': float(np.mean(raw)) if raw else None,
        'conditional_95_interval': np.quantile(estimates, [.025, .975]).tolist() if estimates else None,
        'bootstrap_draws': samples, 'valid_draws': len(estimates), 'zero_case_draws': empty,
        'full_target_dates_per_period': [len(grid) for grid in grids],
        'method': '2,000 paired circular five-date blocks on complete period calendars; group sums/counts zero on absent dates; ratio of total sums to counts; stocks together, mean seed losses',
        'scope': 'conditional on fitted models; descriptive correlated group analysis, not causal or semantic identification'}


def main():
    analysis = ROOT / 'analysis'
    out = ROOT / 'subgroup_calendar'
    if out.exists():
        raise ValueError('Existing supplement preserved')
    summary = read(analysis / 'summary.json')
    for name, expected in read(analysis / 'completion.json').items():
        assert sha256_file(analysis / name) == expected
    rows = read(analysis / 'rows_by_fold.json')
    records = []
    sources = {}
    for fold_rows in rows:
        path = analysis / f"fold_{fold_rows['fold']}_paired_logits.npz"
        record = dict(np.load(path))
        record['dates'] = np.asarray([r['target_date'] for r in fold_rows['rows']])
        np.testing.assert_array_equal(record['targets'], [r['target'] for r in fold_rows['rows']])
        records.append(record)
        sources[str(path)] = sha256_file(path)
    primary = calculate(records)
    np.testing.assert_allclose(primary['delta_log_loss'], summary['primary']['delta_log_loss'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(primary['conditional_95_interval'], summary['primary']['conditional_95_interval'], rtol=0, atol=1e-12)
    groups = []
    for old in summary['groups']:
        group = calculate(records, old['group'])
        assert group['company_days'] == old['company_days']
        if old['primary']:
            np.testing.assert_allclose(group['delta_log_loss'], old['primary']['delta_log_loss'], rtol=0, atol=1e-12)
        groups.append(group)
    out.mkdir()
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    value = {'utc': datetime.now(timezone.utc).isoformat(), 'primary_reproduction': primary, 'groups': groups,
        'source_logits_sha256': sources, 'script_sha256': sha256_file(__file__),
        'amendment_sha256': sha256_file(ROOT / 'AMENDMENT_subgroup_uncertainty.md'),
        'original_analysis_retained': True}
    write_once(out / 'summary.json', value)
    write_once(out / 'completion.json', {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'full_calendar_subgroup_intervals': groups}, indent=2))


if __name__ == '__main__':
    main()
