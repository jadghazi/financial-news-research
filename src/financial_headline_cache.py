"""Pinned frozen RoBERTa cache for explicitly bounded headline development data.

Reuses the verified resumable encoder; never changes older cache identities.
Encoding all development text does not fit a transform: learned centering and
scaling remain the rolling runner's training-only responsibility.
"""
import hashlib
import json
from pathlib import Path

from src.finin_history_data import _boundary, _time
from src.reproduction_v2_data import sha256_file
from src.reproduction_v2_features import (
    ENCODERS, SENTIMENT, _cache_kind, _read, _write, _verify_models,
)


def collect(cohort_dir):
    cohort_dir = Path(cohort_dir)
    source = cohort_dir / 'devsamples_post_close.jsonl'
    provenance = cohort_dir / 'provenance.json'
    manifest = _read(provenance)
    if manifest.get('include_test') is not False:
        raise ValueError('Explicit development-only source required')
    boundary = _boundary(manifest['development_target_before'])
    items = {}
    for line in source.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['split'] not in {'train', 'valid'} or row['target_date'] >= boundary:
            raise ValueError('Future/held-out target in development cache inputs')
        key = 'description:' + row['ticker']
        if items.setdefault(key, row['description']) != row['description']:
            raise ValueError('Description identity changed')
        cutoff = _time(row['cutoff_utc'])
        for headline in row['headlines']:
            text, key = headline['text'], headline['id']
            if not isinstance(text, str) or not text.strip() or hashlib.sha256(text.encode()).hexdigest() != key:
                raise ValueError('Headline identity is not its actual content hash')
            publication = _time(headline['created_at'])
            available = _time(headline.get('available_at', headline['created_at']))
            if publication > available or available > cutoff:
                raise ValueError('Headline unavailable by declared forecast cutoff')
            if items.setdefault(key, text) != text:
                raise ValueError('Headline content collision')
    if not items: raise ValueError('No development text')
    return sorted(items.items()), [source, provenance]


def build(cohort_dir, out, *, batch_size=16, max_tokens=128, benchmark_size=100):
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (batch_size,max_tokens,benchmark_size)):
        raise ValueError('Positive integer encoding budget required')
    items, sources = collect(cohort_dir)
    out = Path(out)
    identity = dict(encoder=ENCODERS['roberta'], sentiment=SENTIMENT,
                    pooling='last_hidden_state[:,0,:]', l2_normalized=False,
                    max_tokens=max_tokens, batch_size=batch_size, device='cpu', threads=4,
                    sentiment_order=['positive','neutral','negative'], dtype='float32',
                    source_sha256={p.name:sha256_file(p) for p in sources},
                    items_sha256=hashlib.sha256(json.dumps(items,ensure_ascii=False).encode()).hexdigest(),
                    generating_code_sha256=sha256_file(Path(__file__)),
                    encoder_engine_sha256=sha256_file(Path('src/reproduction_v2_features.py')))
    out.mkdir(parents=True,exist_ok=True)
    ids = [key for key,_ in items]
    if (out/'identity.json').exists():
        if _read(out/'identity.json') != identity or _read(out/'index.json') != ids:
            raise ValueError('Immutable cache source/configuration changed')
    else:
        if any(out.iterdir()): raise FileExistsError('Refusing untracked nonempty cache')
        _write(out/'identity.json',identity); _write(out/'index.json',ids)
    if (out/'manifest.json').exists():
        manifest = _read(out/'manifest.json')
        for filename,checksum in manifest['files_sha256'].items():
            if sha256_file(out/filename) != checksum: raise ValueError('Completed cache changed')
        for state in manifest['states'].values(): _verify_models(state['model_provenance'])
        return manifest
    states = {kind:_cache_kind(kind,items,out,identity,'roberta',batch_size,max_tokens,False,benchmark_size)
              for kind in ('text','sentiment')}
    manifest = dict(identity=identity,items=len(items),description_items=sum(k.startswith('description:') for k in ids),
                    message_items=sum(not k.startswith('description:') for k in ids),states=states,
                    files_sha256={name:sha256_file(out/name) for name in
                                  ('identity.json','index.json','text.npy','sentiment.npy','text_state.json','sentiment_state.json')})
    _write(out/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('cohort');parser.add_argument('out')
    args=parser.parse_args();build(args.cohort,args.out)
