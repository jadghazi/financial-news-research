"""Bounded training-only integrity diagnostics; never financial evidence.

Memorizing sixteen observations verifies optimization capability. A fabricated
headline-dependent target with identical prices/sentiment verifies that real
cached text can drive learning. Neither diagnoses market predictability.
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from src.finin_headline_experiment import model_factory
from src.reproduction_v2_data import sha256_file
from src.training import set_seed


def _fit(batch, steps=300):
    set_seed(17)
    model = model_factory('full_concat')
    # Deterministic optimization diagnostic, declared departure from screen.
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0.)
    history = []
    for step in range(steps+1):
        optimizer.zero_grad()
        logits, _ = model(batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch['target'])
        if not torch.isfinite(loss): raise ValueError('Nonfinite learning gate')
        if step in (0, 25, 100, steps):
            history.append(dict(step=step, loss=float(loss.detach()),
                accuracy=float(((logits >= 0) == batch['target'].bool()).float().mean()),
                predicted_down=int((logits < 0).sum())))
        if step == steps: break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    return model, history


def run(dataset, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    # No selection/evaluation observations are accepted.
    if any(r['split'] != 'train' for r in dataset.rows):
        raise ValueError('Training-only diagnostic required')
    selected = []
    for label in (0, 1):
        choices = [i for i,r in enumerate(dataset.rows)
                   if r['target_up'] == label and dataset[i]['mask'].any()]
        if len(choices) < 8: raise ValueError('Insufficient balanced training examples')
        selected.extend(choices[:8])
    batch = default_collate([dataset[i] for i in selected])
    set_seed(17)
    initial = model_factory('full_concat').eval()
    probe = copy.deepcopy(batch)
    probe['headline_vectors'].requires_grad_()
    logits, _ = initial(probe)
    logits.sum().backward()
    gradient = float(probe['headline_vectors'].grad[probe['mask']].norm())
    changed = copy.deepcopy(batch)
    changed['headline_vectors'] = batch['headline_vectors'].flip(0)
    changed['sentiment'] = batch['sentiment'].flip(0)
    changed['mask'] = batch['mask'].flip(0)
    with torch.no_grad():
        text_effect = float((initial(changed)[0]-initial(batch)[0]).abs().max())
        set_seed(17)
        no_news = model_factory('matched_no_news').eval()
        masked_effect = float((no_news(changed)[0]-no_news(batch)[0]).abs().max())
    actual_model, actual = _fit(batch)
    # Actual text, fabricated target. All other information identical across rows.
    fabricated = copy.deepcopy(batch)
    fabricated['market_features'] = torch.zeros_like(batch['market_features'])
    fabricated['sentiment'] = torch.zeros_like(batch['sentiment'])
    fabricated['mask'] = torch.ones_like(batch['mask'])
    for i in range(16):
        active = batch['headline_vectors'][i][batch['mask'][i]]
        fabricated['headline_vectors'][i] = active.mean(0)
    means = fabricated['headline_vectors'][:,0,0].numpy()
    coordinate = int(np.var(means, axis=0).argmax())
    values = means[:,coordinate]
    if len(np.unique(values)) != 16: raise ValueError('Fabricated headline rule has ties')
    threshold = float(np.median(values))
    fabricated['target'] = torch.tensor(values > threshold, dtype=torch.float32)
    artificial_model, artificial = _fit(fabricated)
    with torch.no_grad():
        actual_logits = actual_model(batch)[0].tolist()
        artificial_logits = artificial_model(fabricated)[0].tolist()
    torch.save(actual_model.state_dict(),out/'actual_training_state.pt')
    torch.save(artificial_model.state_dict(),out/'artificial_training_state.pt')
    result = dict(training_ids=[dataset.rows[i]['id'] for i in selected],
        seed=17, optimizer='AdamW .003, no weight decay, dropout disabled, 300 full-batch steps',
        headline_gradient_norm=gradient, changed_news_max_logit_delta=text_effect,
        masked_news_max_logit_delta=masked_effect,
        actual_training_labels=actual, artificial_news_labels=artificial,
        predictions=dict(actual_targets=batch['target'].tolist(),actual_logits=actual_logits,
            artificial_targets=fabricated['target'].tolist(),artificial_logits=artificial_logits),
        state_sha256={name:sha256_file(out/name) for name in
            ('actual_training_state.pt','artificial_training_state.pt')},
        artificial_rule=dict(coordinate=coordinate, threshold=threshold,
            definition='training-batch largest-variance coordinate of mean cached text; balanced median split; identical prices/sentiment/counts'),
        scope='Engineering training-only diagnostic. Artificial targets are not financial outcomes. Memorization is not generalization.',
        source_sha256=sha256_file(Path(__file__)))
    result['passed'] = (gradient > 0 and text_effect > 0 and masked_effect == 0
        and actual[-1]['accuracy'] == 1. and artificial[-1]['accuracy'] == 1.
        and actual[-1]['loss'] < .1 and artificial[-1]['loss'] < .1)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    if not result['passed']: raise RuntimeError('Learning gate failed; evidence retained')
    return result
