"""Paired real-block-LLL / LLL-SP benchmark with resumable, versioned output.

Requires fpylll and Sage only when generating/preprocessing/executing. Use
`python block_bench_cli.py all --config block_bench_smoke.json --output ...`.

The input basis is generated once and independently size-reduced once per
preprocessing configuration. Each scheduler/backend uses the same persisted
basis, with independent state. No parallel or interleaved fplll calls.
"""
from __future__ import annotations

import hashlib
import json
import traceback
from pathlib import Path
from time import perf_counter

from block_bench_core import (
    SCHEMA_VERSION, completed_run, digest, normalized_config, dimension,
    normalize_scheduler, normalize_reason, quality, read_record, run_identity,
    write_record,
)


def _basis_hash(rows):
    from block_bench_core import canonical
    return hashlib.sha256(canonical(rows).encode('utf-8')).hexdigest()


def generate(config, root):
    """Add a plan/instances to the manifest; never discard previous plans.

    Instance IDs include the complete family specification and sample index,
    so changing a modulus/bitsize/k etc. cannot reuse an unrelated lattice.
    """
    cfg = normalized_config(config)
    root = Path(root)
    manifest_file = root / 'manifest.json'
    if manifest_file.exists():
        manifest = read_record(manifest_file)
        if manifest.get('schema_version') != SCHEMA_VERSION or manifest.get('kind') != 'block_lll':
            raise ValueError('manifest is not a block benchmark v1; use a separate output folder')
    else:
        manifest = {'schema_version': SCHEMA_VERSION, 'kind': 'block_lll',
                    'plans': {}, 'instances': {}}
    plan_id = digest({'schema_version': SCHEMA_VERSION, 'config': cfg})
    instance_ids = []
    from bench_generators import generate_one, matrix_to_rows
    for family, specs in cfg['families'].items():
        for spec in (specs if isinstance(specs, list) else [specs]):
            spec = dict(spec)
            count = int(spec.pop('count', 1))
            group_id = digest({'family':family, 'spec':spec})
            for idx in range(count):
                id_source = {'family':family, 'spec':spec, 'index':idx,
                             'master_seed':cfg['seed']}
                instance_id = digest(id_source)
                seed = 1 + (int(digest({'lattice_seed':id_source}, 15), 16) % (2**31-2))
                rel = f'instances/v{SCHEMA_VERSION}/{instance_id}.json.gz'
                path = root / rel
                if path.exists():
                    existing = read_record(path)
                    if (existing.get('schema_version') != SCHEMA_VERSION
                        or existing.get('id_source') != id_source
                        or existing.get('basis_sha256') != _basis_hash(existing['basis'])):
                        raise ValueError(f'corrupt or mismatched cached instance {path}')
                    basis_hash = existing['basis_sha256']
                else:
                    B, meta = generate_one(family, spec, seed)
                    rows = matrix_to_rows(B)
                    if len(rows) != dimension(family,spec) or any(len(row) != len(rows) for row in rows):
                        raise ValueError(f'generator produced a non-square/unexpected basis: {family}, {spec}')
                    basis_hash = _basis_hash(rows)
                    write_record(path, {'schema_version':SCHEMA_VERSION, 'id_source':id_source,
                                        'id':instance_id, 'seed':seed, 'family':family,
                                        'spec':spec, 'meta':meta, 'group_id':group_id,
                                        'basis_sha256':basis_hash, 'basis':rows})
                record = {'id':instance_id, 'family':family, 'group_id':group_id,
                          'spec':spec, 'index':idx, 'seed':seed, 'file':rel,
                          'dimension':dimension(family,spec), 'basis_sha256':basis_hash}
                if instance_id in manifest['instances'] and manifest['instances'][instance_id] != record:
                    raise ValueError(f'instance ID collision for {instance_id}')
                manifest['instances'][instance_id] = record
                instance_ids.append(instance_id)
    plan = {'config':cfg, 'instances':instance_ids}
    if plan_id in manifest['plans'] and manifest['plans'][plan_id] != plan:
        raise ValueError(f'plan ID collision for {plan_id}')
    manifest['plans'][plan_id] = plan
    write_record(manifest_file, manifest)
    print(f'[generate] plan {plan_id}: {len(instance_ids)} instances, schema v{SCHEMA_VERSION}', flush=True)
    return plan_id


def _prepared_basis(root, instance, cfg):
    """Cache the exact size-reduced integer basis shared by both backends."""
    from bench_generators import from_rows, matrix_to_rows
    input_spec = {'schema_version':SCHEMA_VERSION,
                  'instance_id':instance['id'],
                  'basis_sha256':instance['basis_sha256'],
                  'delta':cfg['delta'], 'eta':cfg['eta'],
                  'float_type':'mpfr',
                  'precision':cfg['preprocessing_precision']}
    prep_key = digest(input_spec)
    target = root / f'inputs/v{SCHEMA_VERSION}/{prep_key}.json.gz'
    if target.exists():
        record = read_record(target)
        if record.get('input_spec') != input_spec or record.get('basis_sha256') != _basis_hash(record['basis']):
            raise ValueError(f'corrupt prepared input {target}')
        return prep_key, record

    from fpylll import FPLLL, GSO, LLL
    FPLLL.set_threads(1)
    FPLLL.set_precision(int(cfg['preprocessing_precision']))
    raw = read_record(root / instance['file'])
    if raw['basis_sha256'] != instance['basis_sha256'] or _basis_hash(raw['basis']) != instance['basis_sha256']:
        raise ValueError(f'input basis changed on disk for {instance["id"]}')
    B = from_rows(raw['basis'])
    M = GSO.Mat(B, float_type='mpfr')
    if not M.update_gso():
        raise ArithmeticError('preprocessing GSO failed')
    t0 = perf_counter()
    L = LLL.Reduction(M, delta=float(cfg['delta']), eta=float(cfg['eta']))
    L.size_reduction(0, B.nrows, 0)
    if not M.update_gso():
        raise ArithmeticError('size-reduced input GSO failed')
    rows = matrix_to_rows(B)
    record = {'schema_version':SCHEMA_VERSION, 'input_spec':input_spec,
              'basis_sha256':_basis_hash(rows), 'basis':rows,
              'preprocessing_seconds':perf_counter()-t0}
    write_record(target, record)
    return prep_key, record


def _job_specs(instance, prep_key, cfg):
    """Generate identities without accessing fpylll; same seeds across schedules."""
    for cond in cfg['conditions']:
        for sched in cfg['schedulers']:
            schedule_spec = {'name':sched} if isinstance(sched, str) else dict(sched)
            for backend in ('real', 'sim'):
                num = 1 if backend == 'real' else int(cfg['simulation_seeds_per_lattice'])
                for j in range(num):
                    seed = None if backend == 'real' else (1 + int(digest({
                        'seed_instance':instance['id'], 'condition':cond,
                        'replicate':j, 'master_seed':cfg['seed']}, 15), 16) % (2**31-2))
                    identity = run_identity(instance, prep_key, cond, schedule_spec,
                                            cfg, backend, seed)
                    job_key = digest(identity)
                    yield {'identity': identity, 'job_key':job_key,
                           'condition':cond, 'scheduler':schedule_spec,
                           'backend':backend, 'seed':seed}


def _runtime_result(result, scheduler, schedule_name, measured_seconds):
    profile = result.final_profile
    records = []
    for rec in result.records:
        w = rec.window
        records.append({'tour':rec.tour, 'index':rec.index,
                        'start':w.start, 'stop':w.stop, 'phase':w.phase,
                        'label':w.label, 'seconds':rec.seconds})
    nblocks = result.block_reductions
    block_sec = sum(r['seconds'] for r in records)
    return {
        'reason': normalize_reason(result, scheduler, schedule_name),
        'tours':result.tours, 'blocks':result.block_reductions,
        'wall_seconds':result.wall_seconds,
        'invocation_seconds':measured_seconds,
        'block_reduction_seconds':block_sec,
        'overhead_seconds':max(0., result.wall_seconds-block_sec),
        'wall_seconds_per_block':result.wall_seconds/nblocks if nblocks else None,
        'local_seconds_per_block':block_sec/nblocks if nblocks else None,
        'metrics':quality(profile),
        'final_profile_ln_gs':profile,
        'blocks_trace':records,
    }


def _execute(prepared, cfg, job):
    from bench_generators import from_rows
    from fpylll import FPLLL, GSO
    B = from_rows(prepared['basis'])
    condition = job['condition']
    sched_spec = job['scheduler']
    scheduler = normalize_scheduler(sched_spec, n=B.nrows, delta=float(cfg['delta']),
                                    eta=float(cfg['eta']), limits=cfg['limits'])
    backend = job['backend']
    t0 = perf_counter()
    if backend == 'real':
        from real_block_lll import run_real_block_lll
        opts = cfg['real']
        result = run_real_block_lll(
            B, scheduler, delta=float(cfg['delta']), eta=float(cfg['eta']),
            condition=condition, float_type=opts.get('float_type','mpfr'),
            precision=int(opts.get('precision',212)), copy_basis=False,
            record_blocks=True)
    elif backend == 'sim':
        from sandpile_mpfr import SandpileSeq
        from block_lllsp_estimator import run_block_estimator
        opts = cfg['simulator']
        fptype = opts.get('fptype','mpfr')
        if fptype == 'mpfr':
            FPLLL.set_precision(int(opts.get('precision',212)))
        M = GSO.Mat(B, float_type=fptype)
        if not M.update_gso():
            raise ArithmeticError('simulation initial GSO failed')
        kw = {'condition':condition, 'delta':float(cfg['delta']),
              'eta':float(cfg['eta']), 'seed':job['seed'],
              'fptype':fptype, 'allow_lossy_gso':False, 'trace_swaps':False}
        if fptype == 'mpfr':
            kw['precision'] = int(opts.get('precision',212))
        sim = SandpileSeq(M, **kw)
        result = run_block_estimator(
            sim, scheduler, max_swaps_per_block=int(opts.get('max_swaps_per_block',0x400000000)),
            resample_external=bool(opts.get('resample_external',True)),
            record_blocks=True)
    else:
        raise ValueError(backend)
    return _runtime_result(result, scheduler, sched_spec['name'], perf_counter()-t0)


def run(config, root, *, force=False, max_jobs=None):
    """Run a generated plan, resuming successful jobs; failed jobs are retried.

    Jobs execute sequentially so global fplll precision is never raced. Results
    are immutable for each job key unless force=True. Returns a count summary.
    """
    root = Path(root)
    cfg = normalized_config(config)
    plan_id = digest({'schema_version':SCHEMA_VERSION, 'config':cfg})
    manifest = read_record(root/'manifest.json')
    if manifest.get('schema_version') != SCHEMA_VERSION or plan_id not in manifest.get('plans',{}):
        raise ValueError('unknown plan; run `generate` first using the SAME config')
    done, skipped, failed = 0, 0, 0
    attempts = 0
    for instance_id in manifest['plans'][plan_id]['instances']:
        inst = manifest['instances'][instance_id]
        # Generating the cached preprocessing input is intentionally separated
        # from any individual reduction's runtime.
        prep_key, prepared = _prepared_basis(root, inst, cfg)
        prepared_sha = prepared['basis_sha256']
        for job in _job_specs(inst, prep_key, cfg):
            rel = f'runs/v{SCHEMA_VERSION}/{inst["group_id"]}/{inst["id"]}/{job["job_key"]}.json.gz'
            path = root / rel
            if not force and completed_run(path, job['job_key']):
                # Early v1 results omitted this additive metadata field.
                # Backfill it from the validated, cached prepared input without
                # rerunning native reductions or changing the result identity.
                cached = read_record(path)
                if 'prepared_basis_sha256' not in cached:
                    cached['prepared_basis_sha256'] = prepared_sha
                    write_record(path, cached)
                elif cached['prepared_basis_sha256'] != prepared_sha:
                    raise ValueError(f'prepared basis checksum mismatch in {path}')
                skipped += 1
                continue
            if max_jobs is not None and attempts >= max_jobs:
                return {'plan':plan_id, 'finished':done, 'skipped':skipped,
                        'failed':failed, 'limited':True}
            attempts += 1
            result = {
                'schema_version':SCHEMA_VERSION, 'kind':'block_lll_run',
                'job_key':job['job_key'], 'identity':job['identity'],
                'family':inst['family'], 'group_id':inst['group_id'],
                'dimension':inst['dimension'], 'lattice_spec':inst['spec'],
                'instance_id':inst['id'], 'preprocessing_key':prep_key,
                'prepared_basis_sha256':prepared_sha,
                'preprocessing_seconds':prepared.get('preprocessing_seconds'),
                'condition':job['condition'], 'scheduler':job['scheduler'],
                'backend':job['backend'], 'seed':job['seed'],
            }
            try:
                result.update(_execute(prepared, cfg, job))
                result['status'] = 'finished'
                done += 1
                desc = f'RHF={result["metrics"]["rhf"]:.8f}, blocks={result["blocks"]}'
            except Exception as exc:
                result.update({'status':'failed', 'error':repr(exc),
                               'traceback':traceback.format_exc()})
                failed += 1
                desc = f'FAILED: {exc!r}'
            write_record(path, result)
            print(f'[run] {inst["family"]} {inst["id"]} {job["condition"]} '
                  f'{job["scheduler"]["name"]} {job["backend"]} '
                  f'seed={job["seed"]}: {desc}', flush=True)
    return {'plan':plan_id, 'finished':done, 'skipped':skipped,
            'failed':failed, 'limited':False}


def list_jobs(config, root):
    """Enumerate expected output paths without running fpylll/Sage.

    Preprocessing keys are computed identically to `_prepared_basis`.
    """
    cfg = normalized_config(config)
    plan_id = digest({'schema_version':SCHEMA_VERSION, 'config':cfg})
    manifest = read_record(Path(root)/'manifest.json')
    if plan_id not in manifest['plans']:
        raise ValueError('run generate first')
    for instance_id in manifest['plans'][plan_id]['instances']:
        inst = manifest['instances'][instance_id]
        prep_key = digest({'schema_version':SCHEMA_VERSION,
                           'instance_id':inst['id'], 'basis_sha256':inst['basis_sha256'],
                           'delta':cfg['delta'], 'eta':cfg['eta'],
                           'float_type':'mpfr','precision':cfg['preprocessing_precision']})
        for job in _job_specs(inst, prep_key, cfg):
            path = (Path(root)/f'runs/v{SCHEMA_VERSION}/{inst["group_id"]}/'
                    f'{inst["id"]}/{job["job_key"]}.json.gz')
            yield inst, job, path
