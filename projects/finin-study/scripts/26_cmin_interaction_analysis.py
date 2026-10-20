"""Bounded meeting analysis: saved tensors/predictions only; no model fitting.

See docs/code_guide.md#pipeline-entry-points for the code map, contracts, and execution boundaries.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cmin_panel_experiment import paired_uncertainty, score

BASE = ROOT / 'artifacts/cmin_panel_2026-10-14_v1'
NEW = ROOT / 'artifacts/cmin_interaction_2026-10-14_v1'
OUT = ROOT / 'artifacts/diagnostics/cmin_interactions_meeting_2026-10-14_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=True)
        stream.write('\n')


def trace_audit():
    if OUT.exists():
        raise ValueError('Existing audit preserved; use its saved result')
    OUT.mkdir(parents=True)
    write(OUT / 'PROTOCOL.json', {'protocol': str(ROOT / 'docs/cmin_interaction_protocol_2026-10-14.md'),
          'protocol_sha256': digest(ROOT / 'docs/cmin_interaction_protocol_2026-10-14.md'),
          'scope': 'post-inspection exploratory saved-tensor audit; no new inference or fitting'})
    sources, buckets, models, example = {}, [], [], None
    items = read(BASE / 'items.json')
    daily = np.load(BASE / 'daily_arrays.npz')
    for fold in range(2):
        for seed in (17, 42):
            folder = BASE / f'fold_{fold}/finin_{seed}'
            paths = [folder / name for name in ('trace.npz', 'trace_rows.json', 'selected.pt', 'summary.json')]
            sources.update({str(p.relative_to(ROOT)): digest(p) for p in paths})
            summary = read(folder / 'summary.json')
            assert digest(folder / 'selected.pt') == summary['files_sha256']['selected.pt']
            rows, a = read(folder / 'trace_rows.json'), np.load(folder / 'trace.npz')
            state = {k: v.numpy() for k, v in torch.load(folder / 'selected.pt', weights_only=True).items()}
            values = a['news_encoded'] @ state['self_value.weight'].T + state['self_value.bias']
            mixed = np.einsum('btij,btjd->btid', a['self_attention_weights'][:, :, 0], values)
            rebuilt = mixed @ state['self_output.weight'].T + state['self_output.bias']
            rebuilt = np.where(a['mask'][..., None], rebuilt, 0)
            np.testing.assert_allclose(rebuilt, a['news_refined'], atol=1e-5, rtol=2e-5)
            rebuilt_summary = np.sum(a['weights'][..., None] * a['news_refined'], axis=2)
            np.testing.assert_allclose(rebuilt_summary, a['day_news_summaries'], atol=1e-6, rtol=1e-6)
            np.testing.assert_array_equal(a['mlp_input'], np.concatenate(
                [a['market_encoded'].reshape(12, -1), a['day_news_summaries'].reshape(12, -1)], axis=1))
            own = []
            for b, row in enumerate(rows):
                for t in range(5):
                    mask = a['mask'][b, t]
                    n = int(mask.sum())
                    if n < 2:
                        continue
                    w = a['self_attention_weights'][b, t, 0][np.ix_(mask, mask)].astype(float)
                    q = a['weights'][b, t, mask].astype(float)
                    np.testing.assert_allclose(w.sum(1), 1, atol=2e-7)
                    np.testing.assert_allclose(q.sum(), 1, atol=2e-7)
                    result = {'fold': fold, 'seed': seed, 'ticker': row['ticker'],
                              'forecast_date': row['forecast_date'], 'history_slot': t, 'headlines': n}
                    for name, key in (('before', 'news_encoded'), ('after', 'news_refined')):
                        x = a[key][b, t, mask].astype(float)
                        lengths = np.linalg.norm(x, axis=1)
                        similarity = (x @ x.T) / np.maximum(lengths[:, None] * lengths[None, :], 1e-30)
                        result[name + '_pairwise_cosine'] = float(similarity[np.triu_indices(n, 1)].mean())
                        result[name + '_length_cv'] = float(lengths.std() / max(lengths.mean(), 1e-30))
                    result['self_normalized_entropy'] = float(np.clip(
                        (-np.sum(w * np.log(np.maximum(w, 1e-30)), axis=1) / np.log(n)).mean(), 0, 1))
                    result['query_normalized_entropy'] = float(np.clip(-np.sum(q * np.log(np.maximum(q, 1e-30))) / np.log(n), 0, 1))
                    own.append(result)
                    buckets.append(result)
            models.append({'fold': fold, 'seed': seed, 'buckets': len(own),
                'forecast_dates': sorted({r['forecast_date'] for r in rows}),
                'maximum_refined_reconstruction_error': float(np.max(np.abs(rebuilt - a['news_refined']))),
                **{'median_' + key: float(np.median([v[key] for v in own])) for key in
                   ('before_pairwise_cosine', 'after_pairwise_cosine', 'after_length_cv',
                    'self_normalized_entropy', 'query_normalized_entropy')}})
            if fold == 0 and seed == 17:
                b = next(i for i, row in enumerate(rows) if row['ticker'] == 'AAPL')
                row = rows[b]
                ids = daily['news_indices'][row['stock_index'], row['session_index']]
                topics = ['Mixed market/Apple mention', 'Broad market outlook', 'Chip-industry topic',
                          'Broad market outlook', 'Other companies in title', 'Direct Apple event']
                example = {'row': row, 'probability': float(1 / (1 + np.exp(-float(a['logits'][b])))),
                    'source_date_proxy_age_hours': a['headline_age_hours'][b, 4].tolist(),
                    'title_audit_rule': 'Title-only topical annotation; no article bodies inspected, not a financial relevance ground truth',
                    'headlines': [{'id': items[int(idx)][0], 'title': items[int(idx)][1],
                        'topic_annotation': topics[k], 'contextual_weight': float(a['weights'][b, 4, k])}
                        for k, idx in enumerate(ids) if idx >= 0],
                    'self_attention_matrix': a['self_attention_weights'][b, 4, 0].tolist()}
    write(OUT / 'buckets.json', buckets)
    write(OUT / 'summary.json', {'models': models, 'example': example,
        'scope': 'Four selected full baseline networks, two first-evaluation snapshot dates; correlated buckets; direction similarity does not measure semantic preservation or vector magnitude',
        'sources_sha256': sources, 'analysis_script_sha256': digest(Path(__file__))})
    print(json.dumps({'models': models, 'example_probability': example['probability']}, indent=2))


def comparison():
    if (NEW / 'interaction_analysis.json').exists():
        raise ValueError('Existing comparison preserved')
    frames, pairs, scores = [], [], []
    for fold in range(2):
        old, new = BASE / f'fold_{fold}', NEW / f'fold_{fold}'
        assert read(BASE / 'plans.json') == read(NEW / 'plans.json')
        assert read(old / 'scaler.json') == read(new / 'scaler.json')
        predictions = {'full': [], 'no_interaction': [], 'prices': [], 'system_full': [], 'system_no_interaction': []}
        frame_rows = None
        for seed in (17, 42):
            fs = [old / f'finin_{seed}', new / f'no_interaction_{seed}', old / f'price_only_{seed}']
            old_state = torch.load(fs[0] / 'initial.pt', weights_only=True)
            new_state = torch.load(fs[1] / 'initial.pt', weights_only=True)
            assert old_state.keys() == new_state.keys()
            assert all(torch.equal(old_state[k], new_state[k]) for k in old_state)
            for name, folder in zip(('full', 'no_interaction', 'prices'), fs):
                saved = read(folder / 'summary.json')
                assert all(digest(folder / f) == h for f, h in saved['files_sha256'].items())
                with (folder / 'predictions.csv').open(encoding='utf-8', newline='') as stream:
                    rows = list(csv.DictReader(stream))
                identity = [{k: v for k, v in row.items() if k != 'logit'} for row in rows]
                if frame_rows is None:
                    frame_rows = identity
                assert identity == frame_rows
                z = np.load(folder / 'selected_logits.npz')['evaluation']
                np.testing.assert_allclose(z, [float(row['logit']) for row in rows], atol=0, rtol=0)
                predictions[name].append(z)
                if name != 'prices':
                    p = saved['training_prior']
                    predictions['system_' + name].append(np.full(len(z), np.log(p / (1 - p)))
                        if saved['constant_fallback_selected_on_validation'] else z)
                scores.append({'fold': fold, 'seed': seed, 'arm': name,
                    'selected_epoch': saved['selected_epoch'], 'constant_fallback': saved['constant_fallback_selected_on_validation'],
                    'raw': score([int(row['target']) for row in rows], z)})
            y = np.array([int(row['target']) for row in rows])
            diff = np.logaddexp(0, predictions['no_interaction'][-1]) - y * predictions['no_interaction'][-1]
            diff -= np.logaddexp(0, predictions['full'][-1]) - y * predictions['full'][-1]
            pairs.append({'fold': fold, 'seed': seed, 'no_interaction_minus_full': float(diff.mean())})
        frame_rows = [{**row, 'target': int(row['target'])} for row in frame_rows]
        frames.append({'fold': fold, 'rows': frame_rows, 'logits': predictions})
    contrasts = [paired_uncertainty(frames, first, second) for first, second in
        [('no_interaction', 'full'), ('no_interaction', 'prices'), ('system_no_interaction', 'system_full')]]
    metrics = {}
    y = [row['target'] for f in frames for row in f['rows']]
    for arm in frames[0]['logits']:
        m = [score(y, np.concatenate([f['logits'][arm][si] for f in frames])) for si in range(2)]
        metrics[arm] = {'mean_seed_log_loss': float(np.mean([v['log_loss'] for v in m])), 'seed_metrics': m}
    unchanged = 0
    for p, expected in read(NEW / 'prior_evidence_hashes.json').items():
        assert digest(ROOT / p) == expected
        unchanged += 1
    write(NEW / 'interaction_analysis.json', {'metrics': metrics, 'comparisons': contrasts,
        'seed_period_differences': pairs, 'fits': scores, 'baseline_files_unchanged': unchanged,
        'verified_initial_pairs': 4, 'evaluation_company_days': len(y),
        'scope': 'Exploratory matched 160-input procedures; self-attention parameters unused in ablation, active capacity differs; conditional intervals omit training uncertainty',
        'analysis_script_sha256': digest(Path(__file__))})
    print(json.dumps({'metrics': {k: v['mean_seed_log_loss'] for k, v in metrics.items()},
                      'comparisons': contrasts, 'pairs': pairs, 'unchanged': unchanged}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['trace', 'comparison'])
    args = parser.parse_args()
    trace_audit() if args.stage == 'trace' else comparison()
