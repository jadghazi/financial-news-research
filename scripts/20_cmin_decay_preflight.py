"""Freeze a fresh decay study; verify/reuse sources without training or encoding."""
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.cmin_panel_data import read, rolling_plans, write_once
from src.cmin_panel_experiment import sha256_file, fit_scaler, make_model, PanelDataset


def main():
    base = Path('artifacts/cmin_panel_2026-10-14_v1')
    config = read('configs/cmin_decay_48h_v1.json')
    out = Path('artifacts') / config['id']
    if out.exists():
        raise RuntimeError('Fresh study namespace already exists; refusing overwrite')
    for name, expected in read(base / 'completion.json').items():
        assert sha256_file(base / name) == expected, name
    for fit in read(base / 'summary.json')['fits']:
        folder = base / f"fold_{fit['fold']}" / f"{fit['arm']}_{fit['seed']}"
        for name, expected in fit['files_sha256'].items():
            assert sha256_file(folder / name) == expected, name
    for name, expected in read(base / 'auxiliary_audit_hashes.json')['files_sha256'].items():
        assert sha256_file(base / name) == expected, name
    rows = read(base / 'rows.json')
    plans = rolling_plans(rows, config['windows'])
    assert plans == read(base / 'plans.json')
    arrays = dict(np.load(base / 'daily_arrays.npz'))
    text = np.load(base / 'features/text.npy', mmap_mode='r')
    sentiment = np.load(base / 'features/sentiment.npy', mmap_mode='r')
    torch.set_num_threads(4)
    checks = []
    for plan in plans:
        fold = plan['fold']
        scaler = fit_scaler(rows, plan['train'], arrays, text, 5)
        # JSON serializes tuple day keys as lists; compare their saved representation.
        assert json.loads(json.dumps(scaler)) == read(base / f'fold_{fold}/scaler.json')
        prior = float(np.mean([rows[i]['target'] for i in plan['train']]))
        ds = PanelDataset(rows, plan['train'][:24], arrays, text, sentiment, scaler, 5)
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=24)))
        for seed in config['seeds']:
            torch.manual_seed(seed)
            a = make_model(config, 'pooled_uniform', prior).eval()
            torch.manual_seed(seed)
            b = make_model(config, 'pooled_decay', prior).eval()
            assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
            with torch.inference_mode():
                ta, tb = a.trace(batch), b.trace(batch)
            torch.testing.assert_close(ta['mlp_input'][:, :80], tb['mlp_input'][:, :80], rtol=0, atol=0)
            torch.testing.assert_close(ta['logits'].sigmoid(), torch.full_like(ta['logits'], prior))
            assert torch.isfinite(tb['logits']).all()
            active = batch['mask'].any(-1).sum(-1) > 1
            assert active.any()
            assert (ta['temporal_news_weights'][active] - tb['temporal_news_weights'][active]).abs().max() > 1e-5
            checks.append({'fold': fold, 'seed': seed, 'identical_initial_state': True,
                           'same_market_input': True, 'mlp_width': ta['mlp_input'].shape[-1],
                           'parameter_count': sum(p.numel() for p in a.parameters())})
    out.mkdir()
    protected = {str(p.relative_to(base)): sha256_file(p) for p in base.rglob('*') if p.is_file()}
    write_once(out / 'protected_baseline_hashes.json', protected)
    reused = []
    for plan in plans:
        fold = plan['fold']
        folder = out / f'fold_{fold}'
        folder.mkdir()
        source = base / f'fold_{fold}/linear_prices_identity.json'
        (folder / source.name).write_bytes(source.read_bytes())
        reused.append({'source': str(source), 'destination': str(folder / source.name),
                       'sha256': sha256_file(source)})
    write_once(out / 'preflight.json', {'checks': checks, 'plans_equal_baseline': True,
        'scalers_equal_baseline': True, 'reused_fitted_linear_references': reused,
        'protected_baseline_files': len(protected),
        'config_sha256': sha256_file('configs/cmin_decay_48h_v1.json'),
        'protocol_sha256': sha256_file('docs/cmin_decay_protocol_2026-10-14.md'),
        'preflight_script_sha256': sha256_file(__file__)})
    (out / 'preflight_source.py').write_bytes(Path(__file__).read_bytes())
    (out / 'git_status_before.txt').write_text(subprocess.run(
        ['git', 'status', '--short'], capture_output=True, text=True).stdout, encoding='utf-8')
    (out / 'tracked_diff_before.patch').write_text(subprocess.run(
        ['git', 'diff', 'HEAD'], capture_output=True, text=True).stdout, encoding='utf-8')
    (out / 'PREPARATION_NOTE.md').write_text(
        'The initial inline preflight wrapper incorrectly treated the auxiliary audit metadata '
        'as a flat file-hash map and attempted to open a file named checks. It stopped before '
        'creating the study directory, changing data or running any fit. This saved preflight '
        'reads files_sha256 correctly and rechecks the evidence. A second preflight check '
        'stopped on tuple-versus-list day-key representations; every numerical transform '
        'was identical, and the saved JSON representation is now compared. Both stops '
        'preceded study-directory creation or financial fitting. Seven relevant existing '
        'model tests passed before fitting. No financial model was trained in preflight.\n', encoding='utf-8')
    print(json.dumps({'preflight': 'passed', 'protected_baseline_files': len(protected),
                      'checks': checks, 'new_neural_fits_ceiling': 8, 'new_convex_fits': 0}, indent=2))


if __name__ == '__main__':
    main()
