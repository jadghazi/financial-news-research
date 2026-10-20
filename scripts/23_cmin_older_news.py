"""Frozen older-news information-set comparison, using the unchanged fit engine.

Prepare/run/analyze are explicit stages. Eight new fits at most; four all-news
fits are reused. No acquisition, encoding, tuning or automatic retry.

See docs/code_guide.md#pipeline-entry-points for the code map, contracts, and execution boundaries.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.cmin_panel_data import read, write_once, rolling_plans
from src.cmin_panel_experiment import (run, make_model, PanelDataset, fit_scaler,
    score, paired_uncertainty, sha256_file, structurally_used_module_parameters)

CONFIG = Path('configs/cmin_older_news_v1.json')
PROTOCOL = Path('docs/cmin_older_news_protocol_2026-10-14.md')


def context():
    cfg = read(CONFIG)
    return cfg, Path('artifacts') / cfg['id'], Path(cfg['data_directory']), Path(cfg['reused_all_news_directory'])


def verify_fit(folder):
    fit = read(folder / 'summary.json')
    for name, expected in fit['files_sha256'].items():
        assert sha256_file(folder / name) == expected, str(folder / name)
    return fit


def verify_completed(folder):
    for name, expected in read(folder / 'completion.json').items():
        assert sha256_file(folder / name) == expected, str(folder / name)
    for fit in read(folder / 'summary.json')['fits']:
        verify_fit(folder / f"fold_{fit['fold']}" / f"{fit['arm']}_{fit['seed']}")


def guards():
    cfg, root, base, full = context()
    identity = read(root / 'identity.json')
    assert identity['config_sha256'] == sha256_file(CONFIG)
    assert identity['protocol_sha256'] == sha256_file(PROTOCOL)
    assert identity['wrapper_sha256'] == sha256_file(__file__)
    for source, expected in identity['shared_sources'].items():
        assert sha256_file(source) == expected, source
    for name, expected in read(root / 'protected_prior_hashes.json').items():
        assert sha256_file(Path(name)) == expected, name
    return cfg, root, base, full


def prepare():
    cfg, root, base, full = context()
    if root.exists():
        raise ValueError('Fresh study directory already exists; preserve existing evidence')
    verify_completed(base)
    verify_completed(full)
    for name, expected in read(base / 'auxiliary_audit_hashes.json')['files_sha256'].items():
        assert sha256_file(base / name) == expected
    final = read(full / 'final_manifest.json')
    for name, expected in final['study_files_sha256'].items():
        assert sha256_file(full / name) == expected, name
    source_identity = read(full / 'study_identity.json')
    for name, expected in source_identity['sources'].items():
        assert sha256_file(name) == expected, name
    original = read(full / 'config.json')
    rows, plans = read(base / 'rows.json'), read(full / 'plans.json')
    assert plans == read(base / 'plans.json') == rolling_plans(rows, original['windows'])
    assert original['seeds'] == [17, 42] and original['history_sessions'] == 5
    assert len(cfg['components']) * len(original['seeds']) * len(plans) == cfg['maximum_new_neural_fits'] == 8
    assert len(cfg['components']) * cfg['maximum_component_neural_seconds'] == cfg['maximum_new_neural_seconds'] == 1200
    components = {}
    for component in cfg['components']:
        component_cfg = {**original, 'id': cfg['id'] + '_' + component['name'],
            'arms': [component['arm']], 'comparison_pairs': [], 'temporal_aggregation': 'pooled',
            'maximum_neural_fits': 4, 'maximum_total_neural_seconds': cfg['maximum_component_neural_seconds'],
            'study_scope': cfg['scope'], 'stop_rule': cfg['stop_rule']}
        component_cfg.pop('half_life_hours', None)
        component_cfg.pop('age_diagnostic', None)
        if component['news_age_sessions'] is None:
            component_cfg.pop('news_age_sessions', None)
        else:
            component_cfg['news_age_sessions'] = component['news_age_sessions']
        components[component['name']] = component_cfg
    arrays = dict(np.load(base / 'daily_arrays.npz'))
    text = np.load(base / 'features/text.npy', mmap_mode='r')
    sentiment = np.load(base / 'features/sentiment.npy', mmap_mode='r')
    torch.set_num_threads(4)
    checks, controls, counts = [], [], []
    for plan in plans:
        fold = plan['fold']
        scaler = fit_scaler(rows, plan['train'], arrays, text, 5)
        assert json.loads(json.dumps(scaler)) == read(full / f'fold_{fold}/scaler.json')
        assert read(full / f'fold_{fold}/scaler.json') == read(base / f'fold_{fold}/scaler.json')
        full_ds = PanelDataset(rows, plan['train'][:24], arrays, text, sentiment, scaler, 5)
        recent_ds = PanelDataset(rows, plan['train'][:24], arrays, text, sentiment, scaler, 5, news_age_sessions=(0, 1))
        all_batch = next(iter(DataLoader(full_ds, batch_size=24)))
        recent_batch = next(iter(DataLoader(recent_ds, batch_size=24)))
        assert not recent_batch['mask'][:, :4].any()
        assert torch.equal(recent_batch['mask'][:, 4], all_batch['mask'][:, 4])
        for key in ('market_features', 'market_description', 'target'):
            torch.testing.assert_close(all_batch[key], recent_batch[key], rtol=0, atol=0)
        for seed in original['seeds']:
            old = torch.load(full / f'fold_{fold}/pooled_uniform_{seed}/initial.pt', weights_only=True)
            prior = float(np.mean([rows[i]['target'] for i in plan['train']]))
            for name, comp_cfg in components.items():
                torch.manual_seed(seed)
                model = make_model(comp_cfg, comp_cfg['arms'][0], prior).eval()
                assert old.keys() == model.state_dict().keys()
                assert all(torch.equal(old[k], v) for k, v in model.state_dict().items())
                with torch.inference_mode():
                    trace = model.trace(recent_batch if name == 'recent' else all_batch)
                assert trace['mlp_input'].shape == (24, 96)
                assert torch.isfinite(trace['logits']).all()
                if name == 'price':
                    assert not trace['mlp_input'][:, 80:].any()
                checks.append({'fold': fold, 'seed': seed, 'component': name,
                    'identical_reused_initial_state': True, 'total_parameters': sum(p.numel() for p in model.parameters()),
                    'structurally_used_module_parameter_upper_bound': structurally_used_module_parameters(model),
                    'mlp_input_values': 96})
        fold_controls = []
        for index in plan['evaluation']:
            row = rows[index]
            si, j = row['stock_index'], row['session_index']
            recent_count = int((arrays['news_indices'][si, j] >= 0).sum())
            older_count = int((arrays['news_indices'][si, j-4:j] >= 0).sum())
            group = ('recent_and_older' if recent_count and older_count else
                     'recent_only' if recent_count else 'older_only' if older_count else 'empty')
            fold_controls.append({'row_index': index, 'ticker': row['ticker'], 'forecast_date': row['forecast_date'],
                'target_date': row['target_date'], 'group': group, 'recent_headlines': recent_count,
                'older_headlines': older_count})
        controls.append({'fold': fold, 'rows': fold_controls})
        counts.append({'fold': fold, 'groups': {group: {
            'company_days': sum(r['group'] == group for r in fold_controls),
            'unique_dates': len({r['target_date'] for r in fold_controls if r['group'] == group})}
            for group in cfg['coverage_groups']}})
    root.mkdir()
    write_once(root / 'config.json', cfg)
    (root / 'PROTOCOL.md').write_bytes(PROTOCOL.read_bytes())
    (root / 'wrapper_source.py').write_bytes(Path(__file__).read_bytes())
    write_once(root / 'plans.json', plans)
    write_once(root / 'row_controls.json', controls)
    write_once(root / 'coverage_before_fitting.json', counts)
    protected = {str(p): sha256_file(p) for folder in (base, full) for p in folder.rglob('*') if p.is_file()}
    write_once(root / 'protected_prior_hashes.json', protected)
    copied = []
    for name, comp_cfg in components.items():
        write_once(root / (name + '_config.json'), comp_cfg)
        for plan in plans:
            fold = plan['fold']
            folder = root / name / f'fold_{fold}'
            folder.mkdir(parents=True)
            source = base / f'fold_{fold}/linear_prices_identity.json'
            dest = folder / source.name
            dest.write_bytes(source.read_bytes())
            copied.append({'source': str(source), 'destination': str(dest), 'sha256': sha256_file(dest)})
    write_once(root / 'identity.json', {'config_sha256': sha256_file(CONFIG),
        'protocol_sha256': sha256_file(PROTOCOL), 'wrapper_sha256': sha256_file(__file__),
        'shared_sources': source_identity['sources'], 'all_news_config_sha256': sha256_file(full / 'config.json'),
        'all_news_completion_sha256': sha256_file(full / 'completion.json'),
        'reused_arm': 'pooled_uniform', 'reused_fits': 4, 'new_fits_ceiling': 8})
    write_once(root / 'preflight.json', {'checks': checks, 'same_price_histories': True,
        'recent_mask_only_current_arrival_session': True, 'same_plans_and_scalers': True,
        'coverage_groups_saved_before_fitting': True, 'copied_linear_references': copied,
        'protected_prior_files': len(protected)})
    (root / 'git_status_before.txt').write_text(subprocess.run(['git', 'status', '--short'], capture_output=True, text=True).stdout, encoding='utf-8')
    (root / 'tracked_diff_before.patch').write_text(subprocess.run(['git', 'diff', 'HEAD'], capture_output=True, text=True).stdout, encoding='utf-8')
    print(json.dumps({'stage': 'prepared', 'protected_prior_files': len(protected),
        'new_fits': 8, 'reused_fits': 4, 'coverage': counts}, indent=2))


def execute():
    cfg, root, base, full = guards()
    summaries = []
    for component in cfg['components']:
        name = component['name']
        comp_cfg = read(root / (name + '_config.json'))
        print(f'component={name}; four fits maximum', flush=True)
        value = run(comp_cfg, base, study_out=root / name, protocol_path=PROTOCOL)
        assert value['completed_neural_fits'] == 4
        assert value['total_neural_seconds'] <= cfg['maximum_component_neural_seconds']
        summaries.append({'component': name, 'fits': 4, 'seconds': value['total_neural_seconds']})
    assert sum(v['fits'] for v in summaries) <= cfg['maximum_new_neural_fits']
    assert sum(v['seconds'] for v in summaries) <= cfg['maximum_new_neural_seconds']
    write_once(root / 'execution.json', {'components': summaries, 'new_neural_fits': 8,
        'reused_neural_fits': 4, 'new_convex_fits': 0, 'total_new_neural_seconds': sum(v['seconds'] for v in summaries),
        'scope': cfg['scope']})
    print(json.dumps(read(root / 'execution.json'), indent=2))


def logloss(y, z):
    return np.logaddexp(0, z) - np.asarray(y) * z


def contrast(records, first, second):
    value = paired_uncertainty(records, first, second)
    pairs = []
    periods = []
    for record in records:
        y = np.asarray([r['target'] for r in record['rows']])
        seed_values = []
        for i, seed in enumerate((17, 42)):
            a = record['logits'][first][i if len(record['logits'][first]) > 1 else 0]
            b = record['logits'][second][i if len(record['logits'][second]) > 1 else 0]
            delta = float(np.mean(logloss(y, a) - logloss(y, b)))
            pairs.append({'fold': record['fold'], 'seed': seed, 'delta_log_loss': delta})
            seed_values.append(delta)
        periods.append({'fold': record['fold'], 'delta_log_loss': float(np.mean(seed_values)),
                        'company_days': len(y), 'unique_dates': len({r['target_date'] for r in record['rows']})})
    return {**value, 'seed_period_differences': pairs, 'period_differences': periods}


def analyze():
    cfg, root, base, full = guards()
    read(root / 'execution.json')
    for name in ('price', 'recent'):
        verify_completed(root / name)
    output = root / 'analysis'
    if output.exists():
        raise ValueError('Existing analysis preserved; refusing overwrite')
    rows, plans = read(base / 'rows.json'), read(root / 'plans.json')
    control_records = read(root / 'row_controls.json')
    records, fits = [], []
    for plan in plans:
        fold = plan['fold']
        eval_rows = [rows[i] for i in plan['evaluation']]
        controls = next(v['rows'] for v in control_records if v['fold'] == fold)
        assert [v['row_index'] for v in controls] == plan['evaluation']
        assert [(v['ticker'], v['target_date']) for v in controls] == [(v['ticker'], v['target_date']) for v in eval_rows]
        y = np.asarray([r['target'] for r in eval_rows], dtype=float)
        prior = float(np.mean([rows[i]['target'] for i in plan['train']]))
        constant = np.full(len(y), np.log(prior / (1 - prior)))
        record = {'fold': fold, 'rows': eval_rows, 'groups': [v['group'] for v in controls],
                  'logits': {'training_prior': [constant], 'equal_probability': [np.zeros(len(y))]}}
        specs = [('price_only', root / 'price', 'price_only'),
                 ('recent_news', root / 'recent', 'pooled_uniform'),
                 ('all_news', full, 'pooled_uniform')]
        for label, directory, stored_arm in specs:
            assert read(directory / 'plans.json') == plans
            assert read(directory / f'fold_{fold}/scaler.json') == read(base / f'fold_{fold}/scaler.json')
            record['logits'][label], record['logits']['system_' + label] = [], []
            for seed in (17, 42):
                folder = directory / f'fold_{fold}/{stored_arm}_{seed}'
                fit = verify_fit(folder)
                z = np.load(folder / 'selected_logits.npz')['evaluation']
                assert len(z) == len(y) and np.isfinite(z).all()
                metrics = score(y, z)
                np.testing.assert_allclose(metrics['log_loss'], fit['evaluation_network']['log_loss'], rtol=0, atol=1e-12)
                record['logits'][label].append(z)
                system = constant.copy() if fit['constant_fallback_selected_on_validation'] else z.copy()
                record['logits']['system_' + label].append(system)
                # Actual training initial-state guard, not only preflight construction.
                initial = torch.load(folder / 'initial.pt', weights_only=True)
                reference = torch.load(full / f'fold_{fold}/pooled_uniform_{seed}/initial.pt', weights_only=True)
                assert initial.keys() == reference.keys() and all(torch.equal(initial[k], reference[k]) for k in initial)
                fits.append({'fold': fold, 'seed': seed, 'arm': label, 'source_folder': str(folder),
                    'reused': label == 'all_news', 'selected_epoch': fit['selected_epoch'],
                    'constant_fallback': fit['constant_fallback_selected_on_validation'],
                    'raw': metrics, 'selected_system': score(y, system), 'parameter_count': fit['parameter_count'],
                    'structurally_used_module_parameter_upper_bound': fit['structurally_used_module_parameters']})
        records.append(record)
    y = np.asarray([r['target'] for record in records for r in record['rows']])
    metrics = {}
    for name in records[0]['logits']:
        n = len(records[0]['logits'][name])
        values = [score(y, np.concatenate([record['logits'][name][i] for record in records])) for i in range(n)]
        metrics[name] = {'mean_seed_log_loss': float(np.mean([v['log_loss'] for v in values])),
                         'mean_seed_accuracy': float(np.mean([v['accuracy'] for v in values])), 'seed_metrics': values}
    pairs = [('all_news', 'recent_news'), ('all_news', 'price_only'), ('recent_news', 'price_only'),
             ('all_news', 'training_prior'), ('recent_news', 'training_prior'), ('price_only', 'training_prior'),
             ('system_all_news', 'system_recent_news')]
    comparisons = [contrast(records, a, b) for a, b in pairs]
    primary = comparisons[0]
    useful_relative = (primary['delta_log_loss'] <= -cfg['meaningful_log_loss_improvement']
        and primary['conditional_95_interval'][1] < 0
        and all(v['delta_log_loss'] < 0 for v in primary['period_differences'])
        and sum(v['delta_log_loss'] < 0 for v in primary['seed_period_differences']) >= 3)
    groups = []
    for group in cfg['coverage_groups']:
        subset = []
        for record in records:
            selected = np.asarray(record['groups']) == group
            if selected.any():
                subset.append({'fold': record['fold'], 'rows': [r for r, keep in zip(record['rows'], selected) if keep],
                    'logits': {name: [z[selected] for z in values] for name, values in record['logits'].items()}})
        n = sum(len(r['rows']) for r in subset)
        groups.append({'group': group, 'company_days': n,
            'unique_dates': len({r['target_date'] for record in subset for r in record['rows']}),
            'primary': contrast(subset, 'all_news', 'recent_news') if n else None,
            'scope': 'descriptive fixed mask group; differing composition, coverage and pooling; not semantic or causal identification'})
    output.mkdir()
    for record in records:
        np.savez(output / f"fold_{record['fold']}_paired_logits.npz", targets=np.asarray([r['target'] for r in record['rows']]),
            groups=np.asarray(record['groups']), **{name: np.stack(values) for name, values in record['logits'].items()})
    write_once(output / 'rows_by_fold.json', [{'fold': v['fold'], 'rows': v['rows']} for v in records])
    result = {'id': cfg['id'], 'utc': datetime.now(timezone.utc).isoformat(),
        'execution': read(root / 'execution.json'), 'evaluation_company_days': len(y),
        'evaluation_unique_dates': len({r['target_date'] for record in records for r in record['rows']}),
        'primary': primary, 'candidate_useful_relative_addition': bool(useful_relative),
        'metrics': metrics, 'comparisons': comparisons, 'groups': groups, 'fits': fits,
        'scope': cfg['scope'], 'baseline_preservation': {'unchanged_files': len(read(root / 'protected_prior_hashes.json'))},
        'source_script_sha256': sha256_file(__file__)}
    write_once(output / 'summary.json', result)
    write_once(output / 'completion.json', {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({'primary': primary, 'candidate_useful_relative_addition': useful_relative,
        'losses': {name: value['mean_seed_log_loss'] for name, value in metrics.items() if not name.startswith('system_')},
        'groups': [{'group': v['group'], 'company_days': v['company_days'],
                    'delta': v['primary']['delta_log_loss'] if v['primary'] else None} for v in groups]}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run', 'analyze'])
    stage = parser.parse_args().stage
    {'prepare': prepare, 'run': execute, 'analyze': analyze}[stage]()
