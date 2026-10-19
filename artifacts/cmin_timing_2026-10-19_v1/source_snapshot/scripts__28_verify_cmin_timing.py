"""Direct frozen-cache and checkpoint replay for the declared timing study."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cmin_timing_study import BASE, OUT, load_inputs, POLICIES
from src.cmin_panel_data import read, write_once
from src.cmin_panel_experiment import PanelDataset, make_model, sha256_file
from src.reproduction_v2_features import load_frozen, _verify_models
from src.research_training import predict_logits


def cache():
    if (OUT / 'cache_reproduction.json').exists():
        raise ValueError('Existing reproduction evidence preserved')
    prior, new = read(BASE / 'features/manifest.json'), read(OUT / 'features/manifest.json')
    items = read(BASE / 'items.json')
    indices = [0, 37, 207, 1042, 2048, 4096, 8192, len(items)-1]
    result = {}
    for kind in ('text', 'sentiment'):
        state = new['states'][kind]
        _verify_models(state['model_provenance'])
        assert state['model_provenance'] == prior['states'][kind]['model_provenance']
        original = np.load(BASE / 'features' / (kind + '.npy'), mmap_mode='r')
        extra = np.load(OUT / 'additional_features' / (kind + '.npy'), mmap_mode='r')
        union = np.load(OUT / 'features' / (kind + '.npy'), mmap_mode='r')
        np.testing.assert_array_equal(union[:len(original)], original)
        np.testing.assert_array_equal(union[len(original):], extra)
        encoder = load_frozen(kind, 'roberta', 64, False)
        values, truncated = encoder.encode([items[i][1] for i in indices])
        np.testing.assert_allclose(values, original[indices], atol=3e-5, rtol=3e-5)
        result[kind] = {'sample_indices': indices, 'maximum_absolute_difference': float(np.max(np.abs(values-original[indices]))),
                        'all_original_values_equal': True, 'all_appended_values_equal': True,
                        'identical_pinned_provenance': True, 'truncated_sample_texts': truncated}
        del encoder
    write_once(OUT / 'cache_reproduction.json', result)
    print(result, flush=True)


def checkpoints():
    if (OUT / 'checkpoint_replay.json').exists():
        raise ValueError('Existing replay evidence preserved')
    torch.set_num_threads(4)
    cfg, rows, plans, text, sentiment = load_inputs()
    records = []
    for policy in POLICIES:
        arrays = dict(np.load(OUT / (policy + '_arrays.npz')))
        for plan in plans:
            fold = plan['fold']
            scaler = read(OUT / f'fold_{fold}/scaler.json')
            ds = PanelDataset(rows, plan['evaluation'], arrays, text, sentiment, scaler, 5)
            for seed in cfg['seeds']:
                folder = BASE / f'fold_{fold}/finin_{seed}' if policy == 'delay2' else OUT / policy / f'fold_{fold}/finin_{seed}'
                s = read(folder / 'summary.json')
                state = torch.load(folder / 'selected.pt', weights_only=True)
                initial = torch.load(folder / 'initial.pt', weights_only=True)
                reference_initial = torch.load(BASE / f'fold_{fold}/finin_{seed}/initial.pt', weights_only=True)
                assert all(torch.equal(initial[k], reference_initial[k]) for k in initial)
                epoch = torch.load(folder / f"epoch_{s['selected_epoch']:02d}_selected.pt", weights_only=True)
                assert all(torch.equal(state[k], epoch[k]) for k in state)
                model = make_model(cfg, 'finin', s['training_prior'])
                model.load_state_dict(state)
                actual = predict_logits(model, ds)
                expected = np.load(folder / 'selected_logits.npz')['evaluation']
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
                records.append({'policy': policy, 'fold': fold, 'seed': seed, 'cases': len(actual),
                    'max_logit_replay_error': float(np.max(np.abs(actual-expected))),
                    'initial_states_equal': True, 'selected_epoch_state_equal': True,
                    'selected_checkpoint_sha256': sha256_file(folder / 'selected.pt')})
    write_once(OUT / 'checkpoint_replay.json', {'records': records, 'script_sha256': sha256_file(Path(__file__))})
    print('Verified all16 selected full-model checkpoints against saved predictions', flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['cache', 'checkpoints'])
    args = p.parse_args()
    cache() if args.stage == 'cache' else checkpoints()
