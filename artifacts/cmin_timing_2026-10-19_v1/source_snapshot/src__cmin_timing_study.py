"""Versioned timing-policy study; historical CMIN preparation is unchanged."""
from __future__ import annotations

import bisect
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.cmin_panel_data import content_id, read, write_once
from src.cmin_panel_experiment import PanelDataset, make_model, paired_uncertainty, save_predictions, score, sha256_file
from src.reproduction_v2_features import _cache_kind, _verify_models, ENCODERS, SENTIMENT
from src.research_training import fixed_budget_fit, predict_logits

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/cmin_panel_2026-10-14_v1'
OUT = ROOT / 'artifacts/cmin_timing_2026-10-19_v1'
PROTOCOL = ROOT / 'docs/cmin_timing_protocol_2026-10-19.md'
POLICIES = ('delay2', 'date0', 'delay1', 'clock_ny')


def eligible_time(source_date, source_clock, policy):
    """Explicit unverified timing scenarios; never silently infer a timezone."""
    if policy == 'clock_ny':
        value = datetime.fromisoformat(source_clock)
        if value.tzinfo is not None:
            raise ValueError('Clock scenario expects naive source timestamps')
        return value.replace(tzinfo=ZoneInfo('America/New_York')).astimezone(timezone.utc)
    if policy not in ('delay2', 'delay1', 'date0'):
        raise ValueError('Unknown declared timing scenario')
    days = {'delay2': 2, 'delay1': 1, 'date0': 0}[policy]
    return datetime.fromisoformat(source_date).replace(tzinfo=timezone.utc) + timedelta(days=days)


def selected_buckets(stories, closes, policy, cap):
    buckets = [[] for _ in closes]
    for story in stories:
        value = eligible_time(story['source_date'], story['source_time_naive'], policy)
        index = bisect.bisect_left(closes, value)
        if index < len(closes):
            buckets[index].append({**story, 'available_utc': value.isoformat()})
    return [sorted(day, key=lambda s: s['id'])[:cap] for day in buckets]


def prepare():
    if (OUT / 'preparation.json').exists():
        m = read(OUT / 'preparation.json')
        for name, expected in m['files_sha256'].items():
            if sha256_file(OUT / name) != expected:
                raise ValueError('Prepared timing artifact changed')
        return m
    cfg = read(BASE / 'config.json')
    old_items = read(BASE / 'items.json')
    old_map = dict(old_items)
    old = dict(np.load(BASE / 'daily_arrays.npz'))
    calendar = read(BASE / 'calendar.json')['sessions']
    closes = [datetime.fromisoformat(s['close_utc']) for s in calendar]
    all_buckets, union, audit, raw_hashes = {}, dict(old_items), {}, {}
    for ticker in cfg['tickers']:
        earliest, counts = {}, Counter()
        path = BASE / 'raw/news' / (ticker + '.csv')
        raw_hashes[str(path.relative_to(ROOT))] = sha256_file(path)
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for record in csv.DictReader(stream, delimiter='\t'):
                date = record.get('date', '')
                if date >= cfg['development_target_before']:
                    continue
                try:
                    datetime.fromisoformat(date)
                except (ValueError, TypeError):
                    counts['invalid_dates'] += 1
                    continue
                text = ' '.join((record.get('title') or '').split())
                if not text or text.lower() in {'nan', 'none'}:
                    continue
                if record['ticker'] != ticker:
                    raise ValueError('Source ticker mismatch')
                stamp = record.get('time', '')
                eligible_time(date, stamp, 'clock_ny')  # refuse unknown clocks rather than silently excluding rows
                key = content_id(text.casefold())
                value = {'id': content_id(text), 'text': text, 'source_date': date,
                         'source_time_naive': stamp, 'source_link': record.get('link')}
                if key not in earliest or (date, text) < (earliest[key]['source_date'], earliest[key]['text']):
                    earliest[key] = value
        audit[ticker] = {'unique_source_titles': len(earliest), 'counts': dict(counts)}
        all_buckets[ticker] = {}
        for policy in POLICIES:
            days = selected_buckets(earliest.values(), closes, policy, cfg['headlines_per_arrival_session'])
            all_buckets[ticker][policy] = days
            for day in days:
                for story in day:
                    prior = union.setdefault(story['id'], story['text'])
                    if prior != story['text']:
                        raise ValueError('Text hash collision')
    # Preserve the original indices, so its scaler and price-only inputs remain byte-identical.
    additions = sorted((k, v) for k, v in union.items() if k not in old_map)
    listing = old_items + additions
    if len(listing) > cfg['maximum_feature_items']:
        raise ValueError('Feature item ceiling exceeded')
    lookup = {k: i for i, (k, _) in enumerate(listing)}
    outputs = []
    coverage = {}
    for policy in POLICIES:
        arrays = {k: v.copy() for k, v in old.items()}
        arrays['news_indices'].fill(-1)
        arrays['publication_proxy'].fill(0)
        arrays['available_times'].fill(0)
        arrays['clock_ny'] = np.zeros_like(old['available_times'])
        arrays['clock_utc'] = np.zeros_like(old['available_times'])
        records = []
        for si, ticker in enumerate(cfg['tickers']):
            for j, day in enumerate(all_buckets[ticker][policy]):
                for ki, story in enumerate(day):
                    arrays['news_indices'][si, j, ki] = lookup[story['id']]
                    arrays['publication_proxy'][si, j, ki] = datetime.fromisoformat(story['source_date']).replace(tzinfo=timezone.utc).timestamp()
                    arrays['available_times'][si, j, ki] = datetime.fromisoformat(story['available_utc']).timestamp()
                    arrays['clock_ny'][si, j, ki] = eligible_time(story['source_date'], story['source_time_naive'], 'clock_ny').timestamp()
                    arrays['clock_utc'][si, j, ki] = datetime.fromisoformat(story['source_time_naive']).replace(tzinfo=timezone.utc).timestamp()
                    records.append({'ticker': ticker, 'arrival_session': calendar[j]['date'], **story})
        if policy == 'delay2':
            for key in old:
                np.testing.assert_array_equal(arrays[key], old[key], err_msg='Original reconstruction: ' + key)
        path = OUT / (policy + '_arrays.npz')
        if path.exists():
            raise ValueError('Incomplete preparation retained')
        np.savez(path, **arrays)
        outputs.append(path.name)
        with (OUT / (policy + '_stories.jsonl')).open('x', encoding='utf-8') as stream:
            for story in records:
                stream.write(json.dumps(story, ensure_ascii=True) + '\n')
        outputs.append(policy + '_stories.jsonl')
        coverage[policy] = {'selected_occurrences': len(records), 'unique_selected_titles': len({s['id'] for s in records})}
    write_once(OUT / 'items.json', listing)
    write_once(OUT / 'additional_items.json', additions)
    write_once(OUT / 'plans.json', read(BASE / 'plans.json'))
    write_once(OUT / 'rows.json', read(BASE / 'rows.json'))
    for fold in range(2):
        write_once(OUT / f'fold_{fold}/scaler.json', read(BASE / f'fold_{fold}/scaler.json'))
    config = {**cfg, 'id': OUT.name, 'policies': list(POLICIES), 'new_policies': list(POLICIES[1:]),
              'maximum_neural_fits': 12, 'maximum_total_neural_seconds': 1800,
              'maximum_additional_encoding_seconds': 600,
              'shared_preprocessing': 'original delay2 training-only scalers; original feature indices preserved'}
    write_once(OUT / 'config.json', config)
    (OUT / 'PROTOCOL.md').write_bytes(PROTOCOL.read_bytes())
    snapshot = OUT / 'source_snapshot'
    snapshot.mkdir()
    source_names = ['src/cmin_timing_study.py', 'scripts/27_cmin_timing.py', 'src/cmin_panel_experiment.py',
                    'src/cmin_panel_data.py', 'src/research_training.py', 'src/models/research_finin.py',
                    'src/models/paper_finin.py', 'src/reproduction_v2_features.py', 'src/evaluation.py']
    for name in source_names:
        (snapshot / name.replace('/', '__')).write_bytes((ROOT / name).read_bytes())
    provenance = {'git_revision': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
                  'git_dirty_diff': subprocess.run(['git', 'diff'], capture_output=True, text=True).stdout,
                  'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'platform': platform.platform(),
                  'pip_freeze': subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout.splitlines(),
                  'raw_news_sha256': raw_hashes, 'source_sha256': {n: sha256_file(ROOT / n) for n in source_names},
                  'original_prepared_sha256': sha256_file(BASE / 'prepared.json'),
                  'original_feature_manifest_sha256': sha256_file(BASE / 'features/manifest.json'),
                  'protocol_sha256': sha256_file(PROTOCOL)}
    write_once(OUT / 'provenance.json', provenance)
    outputs += ['items.json', 'additional_items.json', 'plans.json', 'rows.json', 'config.json', 'PROTOCOL.md',
                'provenance.json', 'fold_0/scaler.json', 'fold_1/scaler.json']
    result = {'original_reconstruction_exact': True, 'old_items': len(old_items), 'additional_items': len(additions),
              'union_items': len(listing), 'coverage': coverage, 'source_audit': audit,
              'files_sha256': {n: sha256_file(OUT / n) for n in outputs}}
    write_once(OUT / 'preparation.json', result)
    return result


def encode():
    preparation = prepare()
    if (OUT / 'features/manifest.json').exists():
        m = read(OUT / 'features/manifest.json')
        for name, expected in m['files_sha256'].items():
            if sha256_file(OUT / 'features' / name) != expected:
                raise ValueError('Extended features changed')
        return m
    prior = read(BASE / 'features/manifest.json')
    for name, expected in prior['files_sha256'].items():
        if sha256_file(BASE / 'features' / name) != expected:
            raise ValueError('Original features changed')
    for state in prior['states'].values():
        _verify_models(state['model_provenance'])
    extra = OUT / 'additional_features'
    extra.mkdir(exist_ok=True)
    additions = read(OUT / 'additional_items.json')
    identity = {**prior['identity'], 'items_sha256': sha256_file(OUT / 'additional_items.json'),
                'scope': 'only newly selected titles; original cache reused unchanged'}
    write_once(extra / 'identity.json', identity)
    started = time.monotonic()
    states = {}
    for kind in ('text', 'sentiment'):
        states[kind] = _cache_kind(kind, additions, extra, identity, 'roberta', 16, 64, False, 16)
        if time.monotonic() - started > 600:
            raise RuntimeError('Additional encoding ceiling exceeded; partial results retained')
    target = OUT / 'features'
    target.mkdir(exist_ok=True)
    for kind in ('text', 'sentiment'):
        old = np.load(BASE / 'features' / (kind + '.npy'), mmap_mode='r')
        new = np.load(extra / (kind + '.npy'), mmap_mode='r')
        if (target / (kind + '.npy')).exists():
            raise ValueError('Incomplete combined cache retained')
        combined = np.lib.format.open_memmap(target / (kind + '.npy'), mode='w+', dtype=np.float32,
                     shape=(len(old) + len(new), old.shape[1]))
        combined[:len(old)] = old
        combined[len(old):] = new
        combined.flush()
        np.testing.assert_array_equal(combined[:len(old)], old)
        if not np.isfinite(combined).all():
            raise ValueError('Nonfinite union cache')
        if kind == 'sentiment':
            np.testing.assert_allclose(combined.sum(1), 1, atol=1e-5)
    result = {'original_items_retained_exactly': preparation['old_items'], 'additional_items_encoded': len(additions),
              'seconds': time.monotonic() - started, 'states': states,
              'files_sha256': {kind + '.npy': sha256_file(target / (kind + '.npy')) for kind in ('text', 'sentiment')}}
    write_once(target / 'manifest.json', result)
    return result


def load_inputs():
    cfg, rows, plans = read(OUT / 'config.json'), read(OUT / 'rows.json'), read(OUT / 'plans.json')
    assert rows == read(BASE / 'rows.json') and plans == read(BASE / 'plans.json')
    assert all(r['target_date'] < cfg['development_target_before'] for r in rows)
    return cfg, rows, plans, np.load(OUT / 'features/text.npy', mmap_mode='r'), np.load(OUT / 'features/sentiment.npy', mmap_mode='r')


def preflight():
    cfg, rows, plans, text, sentiment = load_inputs()
    records, coverage, example = [], {}, []
    base_arrays = dict(np.load(BASE / 'daily_arrays.npz'))
    for policy in POLICIES:
        arrays = dict(np.load(OUT / (policy + '_arrays.npz')))
        for key in ('prices', 'close_seconds', 'descriptions'):
            np.testing.assert_array_equal(arrays[key], base_arrays[key])
        mode_coverage = []
        for plan in plans:
            fold = plan['fold']
            scaler = read(OUT / f'fold_{fold}/scaler.json')
            assert scaler == read(BASE / f'fold_{fold}/scaler.json')
            ds = PanelDataset(rows, plan['train'][:32], arrays, text, sentiment, scaler, 5)
            old_ds = PanelDataset(rows, plan['train'][:32], base_arrays, text, sentiment, scaler, 5)
            b = next(iter(DataLoader(ds, batch_size=32)))
            old_b = next(iter(DataLoader(old_ds, batch_size=32)))
            for key in ('market_features', 'market_description', 'target'):
                torch.testing.assert_close(b[key], old_b[key], atol=0, rtol=0)
            prior = float(np.mean([rows[i]['target'] for i in plan['train']]))
            for seed in cfg['seeds']:
                torch.manual_seed(seed)
                model = make_model(cfg, 'finin', prior).eval()
                state = torch.load(BASE / f'fold_{fold}/finin_{seed}/initial.pt', weights_only=True)
                assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
                with torch.inference_mode():
                    z = model.trace(b)
                    price_model = make_model(cfg, 'price_only', prior).eval()
                    price_model.load_state_dict(torch.load(BASE / f'fold_{fold}/price_only_{seed}/selected.pt', weights_only=True))
                    torch.testing.assert_close(price_model(b)[0], price_model(old_b)[0], atol=0, rtol=0)
                assert z['mlp_input'].shape == (32, 160) and torch.isfinite(z['logits']).all()
                records.append({'policy': policy, 'fold': fold, 'seed': seed,
                                'initial_state_equal': True, 'price_only_input_output_equal': True})
            for i in plan['evaluation']:
                row = rows[i]; si, j = row['stock_index'], row['session_index']; cutoff = arrays['close_seconds'][j]
                ids = arrays['news_indices'][si, j-4:j+1]; active = ids >= 0
                old_ids = set(base_arrays['news_indices'][si, j-4:j+1].ravel()) - {-1}
                new_ids = set(ids.ravel()) - {-1}
                clocks_ny = arrays['clock_ny'][si, j-4:j+1]
                clocks_utc = arrays['clock_utc'][si, j-4:j+1]
                mode_coverage.append({'fold': fold, 'ticker': row['ticker'], 'forecast_date': row['forecast_date'],
                    'target_date': row['target_date'], 'headlines': int(active.sum()),
                    'newest_bucket_headlines': int(active[-1].sum()),
                    'after_cutoff_if_ny': int(((clocks_ny > cutoff) & active).sum()),
                    'after_cutoff_if_utc': int(((clocks_utc > cutoff) & active).sum()),
                    'identity_jaccard_with_delay2': len(new_ids & old_ids) / max(len(new_ids | old_ids), 1)})
            for i in plan['evaluation'][:12]:
                row = rows[i]
                if row['ticker'] == 'AAPL':
                    example.append({'policy': policy, **row,
                        'source_date_proxy_ages_hours': ((arrays['close_seconds'][row['session_index']] - arrays['publication_proxy'][row['stock_index'],row['session_index']]) / 3600)[arrays['news_indices'][row['stock_index'], row['session_index']] >= 0].tolist()})
        coverage[policy] = mode_coverage
    result = {'checks': records, 'coverage': coverage, 'example': example,
              'scope': 'unchanged market inputs, targets and shared transforms; timezone scenarios are assumptions'}
    write_once(OUT / 'preflight.json', result)
    return result


def run():
    cfg, rows, plans, text, sentiment = load_inputs()
    if not (OUT / 'preflight.json').exists():
        raise ValueError('Preflight must pass before fitting')
    torch.set_num_threads(4)
    summaries, neural_seconds = [], 0.
    for policy in cfg['new_policies']:
        arrays = dict(np.load(OUT / (policy + '_arrays.npz')))
        for plan in plans:
            fold = plan['fold']
            scaler = read(OUT / f'fold_{fold}/scaler.json')
            ds = {role: PanelDataset(rows, plan[role], arrays, text, sentiment, scaler, 5) for role in ('train', 'valid', 'evaluation')}
            prior = float(np.mean([r['target'] for r in ds['train'].rows]))
            for seed in cfg['seeds']:
                folder = OUT / policy / f'fold_{fold}/finin_{seed}'
                if (folder / 'summary.json').exists():
                    s = read(folder / 'summary.json')
                    if any(sha256_file(folder / n) != h for n, h in s['files_sha256'].items()):
                        raise ValueError('Completed financial fit changed')
                    summaries.append(s); neural_seconds += s['seconds']; continue
                if folder.exists():
                    raise ValueError('Incomplete fit retained; no automatic retry')
                if neural_seconds >= cfg['maximum_total_neural_seconds']:
                    raise RuntimeError('Neural time ceiling exhausted')
                folder.mkdir(parents=True)
                write_once(folder / 'attempt.json', {'policy': policy, 'fold': fold, 'seed': seed})
                started = time.monotonic()
                print(f'fit {policy} fold={fold} seed={seed}', flush=True)
                def factory():
                    model = make_model(cfg, 'finin', prior)
                    torch.save(model.state_dict(), folder / 'initial.pt')
                    return model
                def guard(model, datasets):
                    if neural_seconds + time.monotonic() - started > cfg['maximum_total_neural_seconds']:
                        raise RuntimeError('Neural time ceiling reached during fit')
                def progress(record):
                    epoch = record['history']['epoch']
                    write_once(folder / f'epoch_{epoch:02d}.json', {'history': record['history'], 'selected_epoch': record['selected_epoch']})
                    np.savez(folder / f'epoch_{epoch:02d}_logits.npz', **record['logits'])
                    torch.save(record['selected_state'], folder / f'epoch_{epoch:02d}_selected.pt')
                fit = fixed_budget_fit(factory, {k: ds[k] for k in ('train', 'valid')}, seed,
                    epochs=cfg['epochs'], batch_size=cfg['batch_size'], learning_rate=cfg['learning_rate'],
                    optimizer_name='adamw', weight_decay=cfg['weight_decay'], extra_diagnostic=guard,
                    progress_callback=progress)
                z = predict_logits(fit['model'], ds['evaluation'])
                elapsed = time.monotonic() - started; neural_seconds += elapsed
                old = torch.load(BASE / f'fold_{fold}/finin_{seed}/initial.pt', weights_only=True)
                assert all(torch.equal(old[k], v) for k, v in fit['initial_state'].items())
                for name, state in [('selected', fit['selected_state']), ('final', fit['final_state'])]:
                    torch.save(state, folder / (name + '.pt'))
                write_once(folder / 'history.json', fit['history'])
                np.savez(folder / 'selected_logits.npz', evaluation=z, **fit['selected_logits'])
                save_predictions(folder / 'predictions.csv', ds['evaluation'], z)
                y = [r['target'] for r in ds['evaluation'].rows]
                system = np.full(len(z), np.log(prior / (1-prior))) if fit['fallback_to_constant'] else z
                s = {'policy': policy, 'fold': fold, 'seed': seed, 'seconds': elapsed,
                     'selected_epoch': fit['selected_epoch'], 'constant_fallback': fit['fallback_to_constant'],
                     'training_prior': prior, 'raw': score(y, z), 'selected_system': score(y, system),
                     'files_sha256': {p.name: sha256_file(p) for p in folder.iterdir() if p.is_file()}}
                write_once(folder / 'summary.json', s)
                summaries.append(s)
                print(f'done {policy}/{fold}/{seed}: loss={s["raw"]["log_loss"]:.6f}', flush=True)
    result = {'fits': summaries, 'new_neural_fits': len(summaries), 'total_neural_seconds': neural_seconds}
    write_once(OUT / 'training_completion.json', result)
    return result


def analyze():
    cfg, rows, plans, text, sentiment = load_inputs()
    frames, fits, seeds = [], [], []
    all_arms = [*POLICIES, 'price_only', 'training_prior']
    for plan in plans:
        fold = plan['fold']; eval_rows = [rows[i] for i in plan['evaluation']]
        y = np.array([r['target'] for r in eval_rows]); predictions = {n: [] for n in all_arms}
        for seed in cfg['seeds']:
            for policy in [*POLICIES, 'price_only']:
                folder = (BASE / f'fold_{fold}' / (f'price_only_{seed}' if policy == 'price_only' else f'finin_{seed}')) if policy in ('delay2', 'price_only') else OUT / policy / f'fold_{fold}/finin_{seed}'
                saved = read(folder / 'summary.json')
                assert all(sha256_file(folder / n) == h for n, h in saved['files_sha256'].items())
                with (folder / 'predictions.csv').open(encoding='utf-8', newline='') as stream:
                    csv_rows = list(csv.DictReader(stream))
                assert [(r['ticker'], r['forecast_date'], r['target_date'], int(r['target'])) for r in csv_rows] == [(r['ticker'], r['forecast_date'], r['target_date'], r['target']) for r in eval_rows]
                z = np.load(folder / 'selected_logits.npz')['evaluation']
                np.testing.assert_array_equal(z, np.array([float(r['logit']) for r in csv_rows]))
                predictions[policy].append(z)
                history = read(folder / 'history.json')
                best = min(history, key=lambda h: h['eval']['valid']['log_loss'])
                assert saved['selected_epoch'] == best['epoch']
                raw = score(y, z)
                oldmetric = saved.get('raw', saved.get('evaluation_network'))
                assert abs(raw['log_loss'] - oldmetric['log_loss']) < 1e-12
                prior = saved['training_prior']
                fallback = saved.get('constant_fallback', saved.get('constant_fallback_selected_on_validation'))
                system_z = np.full(len(z), np.log(prior/(1-prior))) if fallback else z
                prices = {}
                for ticker in cfg['tickers']:
                    idx = [i for i, r in enumerate(eval_rows) if r['ticker'] == ticker]
                    prices[ticker] = score(y[idx], z[idx])['auc']
                fits.append({'policy': policy, 'fold': fold, 'seed': seed, 'raw': raw,
                    'selected_epoch': saved['selected_epoch'], 'constant_fallback': fallback,
                    'selected_system': score(y, system_z), 'within_company_macro_auc': float(np.mean(list(prices.values())))})
            p = float(np.mean([rows[i]['target'] for i in plan['train']]))
            predictions['training_prior'].append(np.full(len(y), np.log(p/(1-p))))
            for policy in cfg['new_policies']:
                def loss(name):
                    z = predictions[name][-1]
                    return np.logaddexp(0, z)-y*z
                seeds.append({'policy': policy, 'fold': fold, 'seed': seed,
                    'minus_delay2': float(np.mean(loss(policy)-loss('delay2'))),
                    'minus_price_only': float(np.mean(loss(policy)-loss('price_only')))})
        frames.append({'fold': fold, 'rows': eval_rows, 'logits': predictions})
    contrasts = [paired_uncertainty(frames, a, b) for a in cfg['new_policies'] for b in ('delay2', 'price_only', 'training_prior')]
    contrasts.append(paired_uncertainty(frames, 'clock_ny', 'date0'))
    metrics = {}
    y = np.array([r['target'] for f in frames for r in f['rows']])
    for arm in all_arms:
        m = [score(y, np.concatenate([f['logits'][arm][i] for f in frames])) for i in range(2)]
        metrics[arm] = {'mean_seed_log_loss': float(np.mean([v['log_loss'] for v in m])),
                        'mean_seed_accuracy': float(np.mean([v['accuracy'] for v in m])), 'seed_metrics': m}
    decisions = {}
    for policy in cfg['new_policies']:
        contrast = next(c for c in contrasts if c['first'] == policy and c['second'] == 'price_only')
        diffs = [s for s in seeds if s['policy'] == policy]
        period = [float(np.mean([d['minus_price_only'] for d in diffs if d['fold'] == f])) for f in range(2)]
        decisions[policy] = {'practical_news_gain': contrast['delta_log_loss'] <= -.005,
            'conditional_interval_below_zero': contrast['conditional_95_interval'][1] < 0,
            'period_differences_vs_prices': period,
            'seed_period_pairs_favoring_news': sum(d['minus_price_only'] < 0 for d in diffs)}
        decisions[policy]['passes_all_declared_news_criteria'] = (decisions[policy]['practical_news_gain'] and
            decisions[policy]['conditional_interval_below_zero'] and all(v < 0 for v in period) and
            decisions[policy]['seed_period_pairs_favoring_news'] >= 3)
    protected = read(OUT / 'protected_prior_hashes.json')
    assert all(sha256_file(ROOT / p) == expected for p, expected in protected.items())
    pre = read(OUT / 'preflight.json')
    coverage = {}
    for policy, records in pre['coverage'].items():
        coverage[policy] = {'mean_headlines': float(np.mean([r['headlines'] for r in records])),
            'empty_case_fraction': float(np.mean([r['headlines'] == 0 for r in records])),
            'after_cutoff_if_ny_occurrences': sum(r['after_cutoff_if_ny'] for r in records),
            'after_cutoff_if_utc_occurrences': sum(r['after_cutoff_if_utc'] for r in records),
            'cases_after_cutoff_if_ny': sum(r['after_cutoff_if_ny'] > 0 for r in records),
            'cases_after_cutoff_if_utc': sum(r['after_cutoff_if_utc'] > 0 for r in records),
            'total_headline_occurrences': sum(r['headlines'] for r in records),
            'mean_headline_jaccard_with_delay2': float(np.mean([r['identity_jaccard_with_delay2'] for r in records]))}
    result = {'metrics': metrics, 'comparisons': contrasts, 'fits': fits, 'seed_period_differences': seeds,
              'decisions': decisions, 'coverage': coverage, 'protected_prior_files_unchanged': len(protected),
              'evaluation_company_days': len(y), 'evaluation_unique_dates': len({r['target_date'] for f in frames for r in f['rows']}),
              'scope': 'exploratory; fixed original training-only transforms; timing changes cap selection and coverage; no verified publication availability or causal mechanism'}
    write_once(OUT / 'analysis.json', result)
    return result


def main(stage):
    OUT.mkdir(exist_ok=True)
    start = OUT / f'{stage}_started.json'
    if not start.exists():
        write_once(start, {'argv': sys.argv, 'started_utc': datetime.now(timezone.utc).isoformat()})
    try:
        fn = {'prepare': prepare, 'encode': encode, 'preflight': preflight, 'run': run, 'analyze': analyze}[stage]
        result = fn()
        print(json.dumps({k: result[k] for k in ('original_reconstruction_exact', 'additional_items', 'seconds', 'new_neural_fits', 'total_neural_seconds', 'decisions') if k in result}, indent=2), flush=True)
    except Exception:
        write_once(OUT / f'{stage}_failure_{time.time_ns()}.json', {'traceback': traceback.format_exc()})
        raise
