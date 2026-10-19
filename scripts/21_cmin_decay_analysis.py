"""Analyze the frozen decay experiment and declared fixed-model age perturbations.

No fitting, feature encoding, downloads or selection. Requires eight completed
fits; refuses existing output. Preserves raw per-case logits for independent audit.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.cmin_panel_data import read, write_once
from src.cmin_panel_experiment import PanelDataset, make_model, score, paired_uncertainty, sha256_file


def losses(y, logits):
    y, z = np.asarray(y, dtype=float), np.asarray(logits, dtype=float)
    return np.logaddexp(0, z) - y * z


def probability(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def infer_age_masks(model, dataset, history):
    """Change only the news mask; ResearchFININ zeroes masked input channels."""
    original, masked, counts, mass = [], [[] for _ in range(history)], [], []
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0):
            trace = model.trace(batch)
            original.extend(trace['logits'].double().tolist())
            counts.extend(batch['mask'].sum(-1).flip(1).tolist())
            mass.extend(trace['temporal_news_weights'].sum(-1).flip(1).double().tolist())
            for age in range(history):
                altered_mask = batch['mask'].clone()
                altered_mask[:, history - 1 - age, :] = False
                # All prices, headline values, sentiments and age proxies remain identical.
                changed = {**batch, 'mask': altered_mask}
                masked[age].extend(model(changed)[0].double().tolist())
    return (np.asarray(original), np.asarray(masked), np.asarray(counts), np.asarray(mass))


def main():
    started = time.monotonic()
    base = Path('artifacts/cmin_panel_2026-10-14_v1')
    study = Path('artifacts/cmin_decay_48h_2026-10-14_v1')
    output = study / 'analysis'
    if output.exists():
        raise ValueError('Existing analysis retained; refusing overwrite')
    config = read(study / 'config.json')
    result = read(study / 'summary.json')
    if result['completed_neural_fits'] != 8:
        raise ValueError('Declared comparison requires eight completed fits')
    for name, expected in read(study / 'completion.json').items():
        assert sha256_file(study / name) == expected, name
    for name, expected in read(study / 'protected_baseline_hashes.json').items():
        assert sha256_file(base / name) == expected, name
    rows, plans = read(base / 'rows.json'), read(study / 'plans.json')
    assert plans == read(base / 'plans.json')
    arrays = dict(np.load(base / 'daily_arrays.npz'))
    text = np.load(base / 'features/text.npy', mmap_mode='r')
    sentiment = np.load(base / 'features/sentiment.npy', mmap_mode='r')
    output.mkdir()
    (output / 'source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    records, fit_diagnostics, provenance, row_metadata = [], [], {}, []
    history = config['history_sessions']
    for plan in plans:
        fold = plan['fold']
        folder = study / f'fold_{fold}'
        scaler = read(folder / 'scaler.json')
        assert scaler == read(base / f'fold_{fold}/scaler.json')
        dataset = PanelDataset(rows, plan['evaluation'], arrays, text, sentiment, scaler, history)
        y = np.asarray([r['target'] for r in dataset.rows], dtype=float)
        prior = float(np.mean([rows[i]['target'] for i in plan['train']]))
        prior_z = np.full(len(y), np.log(prior / (1 - prior)))
        record = {'fold': fold, 'rows': dataset.rows, 'logits': {
            'training_prior': [prior_z], 'equal_probability': [np.zeros(len(y))],
            'price_context': []}}
        row_metadata.append({'fold': fold, 'rows': dataset.rows})
        for seed in config['seeds']:
            a = torch.load(folder / f'pooled_uniform_{seed}/initial.pt', weights_only=True)
            b = torch.load(folder / f'pooled_decay_{seed}/initial.pt', weights_only=True)
            assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
            record['logits']['price_context'].append(np.load(
                base / f'fold_{fold}/price_only_{seed}/selected_logits.npz')['evaluation'])
        for arm in config['arms']:
            record['logits'][arm] = []
            record['logits'][f'system_{arm}'] = []
            for age in range(history):
                record['logits'][f'{arm}_mask_age_{age}'] = []
            for seed in config['seeds']:
                fit_folder = folder / f'{arm}_{seed}'
                fit = read(fit_folder / 'summary.json')
                for name, expected in fit['files_sha256'].items():
                    assert sha256_file(fit_folder / name) == expected, name
                model = make_model(config, arm, fit['training_prior'])
                model.load_state_dict(torch.load(fit_folder / 'selected.pt', weights_only=True))
                z, altered, count, mass = infer_age_masks(model, dataset, history)
                saved = np.load(fit_folder / 'selected_logits.npz')['evaluation']
                np.testing.assert_allclose(z, saved, rtol=1e-6, atol=1e-6)
                for age in range(history):
                    unaffected = count[:, age] == 0
                    np.testing.assert_allclose(altered[age, unaffected], z[unaffected], rtol=0, atol=1e-6)
                np.testing.assert_allclose(mass.sum(1), (count.sum(1) > 0).astype(float), atol=2e-6)
                name = f'fold_{fold}_{arm}_{seed}.npz'
                np.savez(output / name, original=z, masked_age_logits=altered,
                         headline_counts_by_age=count, temporal_mass_by_age=mass, targets=y)
                provenance[str((fit_folder / 'selected.pt').relative_to(study))] = sha256_file(fit_folder / 'selected.pt')
                record['logits'][arm].append(saved)
                record['logits'][f'system_{arm}'].append(
                    prior_z.copy() if fit['constant_fallback_selected_on_validation'] else saved)
                curves = []
                for age in range(history):
                    record['logits'][f'{arm}_mask_age_{age}'].append(altered[age])
                    curves.append({'age_sessions_since_proxy_arrival': age,
                        'delta_log_loss': float(np.mean(losses(y, altered[age]) - losses(y, z))),
                        'headlines_removed': int(count[:, age].sum()),
                        'cases_with_removed_headlines': int((count[:, age] > 0).sum()),
                        'cases_changed_over_1e_minus_6': int((np.abs(probability(altered[age]) - probability(z)) > 1e-6).sum()),
                        'maximum_absolute_probability_change': float(np.max(np.abs(probability(altered[age]) - probability(z)))),
                        'mean_temporal_news_mass_removed': float(mass[:, age].mean())})
                fit_diagnostics.append({'fold': fold, 'arm': arm, 'seed': seed,
                    'raw': score(y, saved), 'selected_system': score(y, record['logits'][f'system_{arm}'][-1]),
                    'selected_epoch': fit['selected_epoch'],
                    'constant_fallback': fit['constant_fallback_selected_on_validation'],
                    'replay_max_absolute_logit_error': float(np.max(np.abs(z - saved))),
                    'curve': curves})
                print(f'diagnostic fold={fold} {arm}/{seed}: replay verified; all five age masks complete', flush=True)
        records.append(record)
    write_once(output / 'rows_by_fold.json', row_metadata)
    y_all = np.asarray([r['target'] for record in records for r in record['rows']], dtype=float)
    combined = {}
    for arm in records[0]['logits']:
        n = len(records[0]['logits'][arm])
        metrics = [score(y_all, np.concatenate([record['logits'][arm][i] for record in records])) for i in range(n)]
        combined[arm] = {'mean_seed_log_loss': float(np.mean([v['log_loss'] for v in metrics])),
                         'mean_seed_accuracy': float(np.mean([v['accuracy'] for v in metrics])),
                         'mean_seed_brier': float(np.mean([v['brier'] for v in metrics])), 'seed_metrics': metrics}
    primary = paired_uncertainty(records, 'pooled_decay', 'pooled_uniform')
    np.testing.assert_allclose(primary['delta_log_loss'], result['paired_comparisons'][0]['delta_log_loss'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(primary['conditional_95_interval'], result['paired_comparisons'][0]['conditional_95_interval'], rtol=0, atol=1e-12)
    comparisons = [primary, paired_uncertainty(records, 'system_pooled_decay', 'system_pooled_uniform')]
    for arm in config['arms']:
        comparisons.append(paired_uncertainty(records, arm, 'training_prior'))
    period_seed = []
    for record in records:
        y = np.asarray([r['target'] for r in record['rows']], dtype=float)
        for i, seed in enumerate(config['seeds']):
            period_seed.append({'fold': record['fold'], 'seed': seed, 'delta_log_loss': float(np.mean(
                losses(y, record['logits']['pooled_decay'][i]) - losses(y, record['logits']['pooled_uniform'][i])))})
    aggregate_curves = []
    for arm in config['arms']:
        for age in range(history):
            name = f'{arm}_mask_age_{age}'
            overall = paired_uncertainty(records, name, arm)
            per_period = [{'fold': record['fold'], **paired_uncertainty([record], name, arm)} for record in records]
            chosen = [f['curve'][age] for f in fit_diagnostics if f['arm'] == arm]
            # Coverage is seed-invariant; use one seed from each period, not four copies.
            coverage = [f['curve'][age] for f in fit_diagnostics if f['arm'] == arm and f['seed'] == config['seeds'][0]]
            aggregate_curves.append({'arm': arm, 'age_sessions_since_proxy_arrival': age,
                **overall, 'period_comparisons': per_period,
                'seed_fold_deltas': [v['delta_log_loss'] for v in chosen],
                'headlines_removed': sum(v['headlines_removed'] for v in coverage),
                'cases_with_removed_headlines': sum(v['cases_with_removed_headlines'] for v in coverage),
                'case_coverage_fraction': sum(v['cases_with_removed_headlines'] for v in coverage) / len(y_all),
                'mean_temporal_news_mass_removed': float(np.average(
                    [f['curve'][age]['mean_temporal_news_mass_removed'] for f in fit_diagnostics if f['arm'] == arm],
                    weights=[f['raw']['count'] for f in fit_diagnostics if f['arm'] == arm]))})
    output_summary = {'id': config['id'], 'utc': datetime.now(timezone.utc).isoformat(),
        'analysis_seconds': time.monotonic() - started, 'neural_fits': 8, 'new_convex_fits': 0,
        'evaluation_company_days': len(y_all), 'evaluation_unique_dates': result['evaluation_unique_dates'],
        'primary': primary, 'comparisons': comparisons, 'period_seed_differences': period_seed,
        'metrics': combined, 'fits': fit_diagnostics, 'age_sensitivity': aggregate_curves,
        'scope': 'Exploratory inspected development periods; fixed-model mask sensitivity with renormalization; relative recency, not publication diffusion or causal contribution',
        'uncertainty_scope': 'Conditional on saved fits; pointwise age intervals, correlated ages, no selection from curves',
        'baseline_preservation': {'protected_files': len(read(study / 'protected_baseline_hashes.json')), 'all_unchanged': True},
        'selected_checkpoint_hashes': provenance, 'script_sha256': sha256_file(__file__)}
    write_once(output / 'summary.json', output_summary)
    write_once(output / 'completion.json', {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({'primary': primary, 'period_seed_differences': period_seed,
        'log_losses': {arm: value['mean_seed_log_loss'] for arm, value in combined.items()
                       if arm in ['pooled_uniform', 'pooled_decay', 'training_prior', 'system_pooled_uniform', 'system_pooled_decay', 'price_context']},
        'analysis_seconds': output_summary['analysis_seconds']}, indent=2))


if __name__ == '__main__':
    main()
