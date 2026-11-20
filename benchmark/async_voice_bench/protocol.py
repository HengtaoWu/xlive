import json
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def prepare_manifest(directory, episodes, system_id, config, timing_source):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not episodes or not isinstance(system_id, str) or not system_id.strip():
        raise ValueError('Manifest requires episodes and a system ID')
    ids = [ep['episode_id'] for ep in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate manifest episode IDs')
    manifest = {
        'system_id': system_id,
        'timing_source': timing_source,
        'config': config,
        'episode_ids': ids,
        'selection_policy': 'One registered attempt per episode; no best-of retry selection.',
    }
    path = directory / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Existing run manifest differs; use a new output directory')
    elif any(directory.glob('*.json')):
        raise ValueError('Cannot register a manifest over unregistered existing runs')
    else:
        write_json(path, manifest)
    return manifest


def load_registered_runs(directory, episodes):
    directory = Path(directory)
    path = directory / 'manifest.json'
    if not path.is_file():
        raise ValueError('Missing manifest.json: register the intended episodes before running')
    manifest = json.loads(path.read_text())
    ids = manifest['episode_ids']
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Invalid manifest episode IDs')
    for eid in ids:
        if eid not in episodes:
            raise ValueError(f'Unknown episode in manifest: {eid}')
    extras = {p.stem for p in directory.glob('*.json')} - {'manifest'} - set(ids)
    if extras:
        raise ValueError(f'Unexpected run files: {sorted(extras)}')
    runs, pending = {}, {}
    for eid in ids:
        path = directory / (eid + '.json')
        if not path.is_file():
            pending[eid] = 'missing_run'
            continue
        run = json.loads(path.read_text())
        for key in ('system_id', 'timing_source'):
            if run.get(key) != manifest[key]:
                raise ValueError(f'Run/manifest {key} mismatch: {eid}')
        if run.get('episode_id') != eid:
            raise ValueError(f'Run episode ID mismatch: {eid}')
        status = run.get('execution_status')
        if status not in ('complete', 'model_failure', 'infrastructure_failure'):
            raise ValueError(f'Missing/unknown execution status: {eid}')
        if status == 'infrastructure_failure':
            pending[eid] = status
        else:
            runs[eid] = run
    return manifest, runs, pending
