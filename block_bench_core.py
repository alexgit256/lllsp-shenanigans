"""Common, dependency-light configuration, metrics and storage for block LLL benchmarks.

Run schema v1 deliberately does NOT record Git commit or library versions.
Input profiles x[i] are ln(||b_i*||); output quality metrics follow bench_runner.
"""
from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from time import perf_counter

SCHEMA_VERSION = 1


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def digest(obj, length=20):
    return hashlib.sha256(canonical(obj).encode('utf-8')).hexdigest()[:length]


def read_record(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as fh:
            return json.load(fh)
    return json.loads(path.read_text(encoding='utf-8'))


def write_record(path, obj):
    """Atomic replacement. A crashed writer never leaves a half-written job.

    Concurrent writes to the *same* path are not supported; execute each
    deterministic job at most once at a time.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as raw:
            if path.suffix == '.gz':
                with gzip.GzipFile(filename='', mode='wb', fileobj=raw) as zipped:
                    zipped.write(canonical(obj).encode('utf-8'))
            else:
                raw.write((json.dumps(obj, indent=2, sort_keys=True) + '\n').encode('utf-8'))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def quality(x):
    """RHF, log2(first squared GSO), slope, log2 potential from ln GS norms.

    Potential weights n-1-i match the existing bench_runner.py. Use log-RHF
    to avoid cancellation when RHF is close to 1, and keep raw profile.
    """
    x = [float(y) for y in x]
    n = len(x)
    if n < 2 or not all(math.isfinite(v) for v in x):
        raise ValueError('expected at least two finite ln GS lengths')
    mean = math.fsum(x) / n
    center = (n-1)/2
    slope_ln = math.fsum((i-center)*(v-mean) for i, v in enumerate(x)) / (n*(n*n-1)/12)
    log2_rhf = (x[0]-mean)/(n*math.log(2))
    return {
        'dimension': n,
        'rhf': math.exp(math.log(2)*log2_rhf),
        'log2_rhf': log2_rhf,
        'slope': 2*slope_ln/math.log(2),
        'log2_potential': 2*math.fsum((n-1-i)*v for i, v in enumerate(x))/math.log(2),
        'log2_first_r': 2*x[0]/math.log(2),
        'log2_covolume': math.fsum(x)/math.log(2),
    }


def normalize_scheduler(spec, *, n, delta, eta, limits=None):
    """Translate declarative names to real lllsp_schedulers Scheduler instances.

    Does not rewrite or approximate the repository's dyadic or greedy schedule.
    'blaster' and 'flatter' remain the repository's documented schedule proxies.
    """
    from lllsp_schedulers import (Scheduler, Window, BKZScheduler,
                                  DyadicScheduler, FlatterScheduler,
                                  BlasterScheduler, GreedyScheduler)
    if isinstance(spec, str):
        spec = {'name': spec}
    spec = dict(spec)
    name = spec.pop('name')
    common = dict(max_tours=2, max_blocks=None)
    common.update(limits or {})
    for k in ('max_tours', 'max_blocks'):
        if k in spec:
            common[k] = spec.pop(k)
    # Preserve explicit None: it means 'no tour/block limit', not the
    # Scheduler constructor default.
    beta = spec.pop('beta', None)
    if beta is not None:
        beta = int(beta)

    class FullScheduler(Scheduler):
        def _make_windows(self, profile):
            return (Window(0, self.n, label='full'),)

    class NonoverlapScheduler(Scheduler):
        def __init__(self, n, beta, **kw):
            self.beta = beta
            super().__init__(n, **kw)
        def _make_windows(self, profile):
            return tuple(Window(i, min(i+self.beta, self.n), label='nonoverlap')
                         for i in range(0, self.n, self.beta)
                         if min(i+self.beta, self.n)-i >= 2)

    if name == 'full':
        if beta not in (None, n):
            raise ValueError('full means exactly one [0,n) window')
        ans = FullScheduler(n, **common)
    elif name == 'nonoverlap':
        if beta is None or not (2 <= beta <= n):
            raise ValueError('nonoverlap requires beta in [2,n]')
        ans = NonoverlapScheduler(n, beta, **common)
    elif name in ('sliding', 'bkz'):
        if beta is None:
            raise ValueError('sliding requires beta')
        ans = BKZScheduler(n, beta, jump=int(spec.pop('jump', 1)), **common)
    elif name == 'dyadic':
        if beta is not None:
            raise ValueError('dyadic does not accept beta; it changes scale per tour')
        ans = DyadicScheduler(n, max_levels=spec.pop('max_levels', None), **common)
    elif name == 'flatter':
        if beta is not None:
            raise ValueError('flatter uses leaf_size, not beta')
        ans = FlatterScheduler(n, leaf_size=int(spec.pop('leaf_size', min(32, n))), **common)
    elif name == 'blaster':
        if beta is None:
            raise ValueError('blaster requires beta')
        ans = BlasterScheduler(n, beta, **common)
    elif name == 'external':
        # Optional scheduler not shipped in the public repository (e.g.,
        # a locally extracted MetricGreedyScheduler).
        import importlib
        factory = spec.pop('factory')
        module_name, sep, symbol = factory.partition(':')
        if not sep or not module_name or not symbol:
            raise ValueError('external factory must be module:ClassName')
        ctor = getattr(importlib.import_module(module_name), symbol)
        kwargs = dict(spec.pop('kwargs', {}))
        if beta is not None:
            kwargs['beta'] = beta
        ans = ctor(n=n, **kwargs, **common)
        if not isinstance(ans, Scheduler):
            raise TypeError('external scheduler must subclass lllsp_schedulers.Scheduler')
    elif name in ('greedy_lower', 'greedy_upper', 'greedy'):
        if beta is None:
            raise ValueError('greedy requires beta')
        metric = 0 if name == 'greedy_lower' else 1 if name == 'greedy_upper' else int(spec.pop('metric', 0))
        ans = GreedyScheduler(n, beta, delta=delta, eta=eta, metric=metric,
                              stride=int(spec.pop('stride', 1)), **common)
    else:
        raise ValueError(f'unknown scheduler {name!r}')
    if spec:
        raise ValueError(f'unused scheduler configuration {spec!r}')
    return ans


def dimension(family, spec):
    if family == 'ntru':
        return 2*int(spec['degree'])
    if family == 'coppersmith':
        return int(spec.get('degree', 2))*int(spec.get('m', 4)) + int(spec.get('t', 2))
    return int(spec['dimension'])


def normalized_config(cfg):
    cfg = dict(cfg)
    if 'families' not in cfg or not isinstance(cfg['families'], dict) or not cfg['families']:
        raise ValueError('configuration requires a nonempty families dict')
    if not cfg.get('schedulers'):
        raise ValueError('configuration requires at least one scheduler')
    cfg.setdefault('seed', 42)
    cfg.setdefault('delta', .99)
    cfg.setdefault('eta', .51)
    if not (.25 < float(cfg['delta']) <= 1 and .5 <= float(cfg['eta']) < math.sqrt(float(cfg['delta']))):
        raise ValueError('invalid (delta, eta)')
    cfg.setdefault('conditions', ['lovasz', 'siegel'])
    if not cfg['conditions'] or any(c not in ('lovasz', 'siegel') for c in cfg['conditions']):
        raise ValueError('conditions must be lovasz/siegel')
    cfg.setdefault('simulation_seeds_per_lattice', 2)
    if int(cfg['simulation_seeds_per_lattice']) < 1:
        raise ValueError('simulation_seeds_per_lattice must be positive')
    cfg.setdefault('real', {'float_type':'mpfr', 'precision': 212})
    cfg.setdefault('simulator', {'fptype':'mpfr', 'precision':212, 'resample_external':True,
                                  'max_swaps_per_block':0x400000000})
    cfg.setdefault('limits', {'max_tours':2, 'max_blocks':1000})
    cfg.setdefault('preprocessing_precision', max(212, int(cfg['real'].get('precision', 212))))
    for family, specs in cfg['families'].items():
        for spec in (specs if isinstance(specs, list) else [specs]):
            n = dimension(family, spec)
            if n < 2 or int(spec.get('count', 1)) < 1:
                raise ValueError(f'invalid family parameters: {family} {spec}')
            for sched in cfg['schedulers']:
                normalize_scheduler(sched, n=n, delta=float(cfg['delta']),
                                    eta=float(cfg['eta']), limits=cfg['limits'])
    return cfg


def normalize_reason(result, scheduler, name):
    reason = result.reason
    if reason == 'scheduler_stop':
        if scheduler.max_blocks is not None and scheduler.block_reductions >= scheduler.max_blocks:
            return 'block_budget'
        if scheduler.max_tours is not None and scheduler.tour >= scheduler.max_tours:
            return 'tour_budget'
        if name == 'greedy_lower':
            return 'greedy_stopped'
        return 'schedule_complete'
    return reason


def completed_run(path, expected_key):
    if not Path(path).exists():
        return False
    try:
        rec = read_record(path)
        return (rec.get('schema_version') == SCHEMA_VERSION
                and rec.get('job_key') == expected_key
                and rec.get('status') == 'finished')
    except (OSError, EOFError, ValueError, KeyError, TypeError):
        return False


def run_identity(instance, prep_key, condition, scheduler, cfg, backend, seed):
    return {
        'schema_version': SCHEMA_VERSION,
        'instance_id': instance['id'], 'basis_sha256': instance['basis_sha256'],
        'preprocessing_key': prep_key,
        'condition': condition, 'scheduler': scheduler,
        'limits': cfg['limits'], 'delta':cfg['delta'], 'eta':cfg['eta'],
        'backend':backend, 'seed':seed,
        'backend_options': deepcopy(cfg['real'] if backend=='real' else cfg['simulator']),
    }
