"""Process-isolated, paired LLL/sandpile experiments.

Native fpylll LLL.Reduction is the reference. A separately instrumented
adjacent-swap LLL *replay* logs mu immediately before valid swaps via fpylll's
size_reduction and MatGSO operations. It is not asserted to have fplll's exact
internal sequence: both counts and final-profile discrepancy are recorded.
"""
from __future__ import annotations
import json
import math
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from bench_generators import from_rows, read_json_gz, write_json_gz, matrix_to_rows


def _real_field(precision):
    from sage.all import RealField
    return RealField(int(precision))


def _log_profile(profile, R):
    return [R(x).log2() for x in profile]


def metrics(profile, precision):
    """High-precision metrics with base-2 logs of squared GSO lengths."""
    R = _real_field(precision)
    p = _log_profile(profile, R)
    n = len(p)
    avg = sum(p, R(0))/n
    mean_i = R(n-1)/2
    denom = sum((R(i)-mean_i)**2 for i in range(n))
    slope = sum((R(i)-mean_i)*(y-avg) for i,y in enumerate(p)) / denom
    log2_rhf = (p[0]-avg)/(2*n)
    log2_potential = sum(R(n-1-i)*x for i,x in enumerate(p))
    return {
        'rhf': str(R(2)**log2_rhf),
        'log2_rhf': str(log2_rhf),
        'slope': str(slope),
        'log2_potential': str(log2_potential),
        'log2_first_r': str(p[0]),
        'log2_covolume': str(sum(p, R(0))/2),
        'dimension': n,
    }


def profile_distance(real_profile, sim_profile, precision):
    R = _real_field(precision)
    a = _log_profile(real_profile, R)
    b = _log_profile(sim_profile, R)
    if len(a) != len(b):
        raise ValueError('profile lengths differ')
    n = len(a)
    diffs = [x-y for x, y in zip(a, b)]
    return {
        'profile_mae': str(sum((abs(v) for v in diffs), R(0))/n),
        'profile_rmse': str((sum((v*v for v in diffs), R(0))/n).sqrt()),
        'profile_max_abs': str(max(abs(v) for v in diffs)),
        'profile_bias': str(sum(diffs, R(0))/n),  # real minus sim
    }


def _mat_gso(B, prec):
    from fpylll import GSO, FPLLL
    FPLLL.set_precision(int(prec))
    M = GSO.Mat(B, float_type='mpfr')
    if not M.update_gso():
        raise ArithmeticError('initial GSO update failed')
    return M


def _snapshot(M, delta, eta):
    from sandpile_mpfr import SandpileSeq
    s = SandpileSeq(M, condition='lovasz', delta=delta, eta=eta,
                    allow_lossy_gso=False)
    return {'r': s.profile(), 'mu': s.mu_matrix(),
            'subdiag_mu': s.final_subdiagonal_mu()}


def _preprocess_size_reduce(B, prec, delta, eta):
    from fpylll import IntegerMatrix, LLL
    A = IntegerMatrix(B)
    M = _mat_gso(A, prec)
    reducer = LLL.Reduction(M, delta=float(delta), eta=float(eta))
    reducer.size_reduction(0, A.nrows, 0)
    if not M.update_gso():
        raise ArithmeticError('size-reduced input has invalid GSO')
    # Python-accessible get_mu is 53-bit, sufficient only for this diagnostic.
    max_mu = max(abs(M.get_mu(i, j)) for i in range(1, A.nrows)
                 for j in range(i))
    if max_mu > float(eta) + 1e-6:
        raise ArithmeticError(f'input not size reduced: max |mu|={max_mu}, eta={eta}')
    return A


def _native_lll(B0, condition, prec, delta, eta):
    from fpylll import LLL, IntegerMatrix
    B = IntegerMatrix(B0)
    M = _mat_gso(B, prec)
    flags = LLL.SIEGEL if condition == 'siegel' else LLL.DEFAULT
    L = LLL.Reduction(M, delta=float(delta), eta=float(eta), flags=flags)
    start = time.perf_counter()
    L()
    runtime = time.perf_counter()-start
    if not M.update_gso():
        raise ArithmeticError('native LLL GSO update failed')
    state = _snapshot(M, delta, eta)
    return {'profile': state['r'], 'mu': state['mu'],
            'final_subdiagonal_mu': state['subdiag_mu'],
            'metrics': metrics(state['r'], prec),
            'swaps': int(L.nswaps), 'runtime_sec': runtime}


def _trace_before_swaps(B0, condition, prec, delta, eta, max_swaps):
    """Instrumented classical *adjacent* LLL using fpylll size reductions.

    fplll's native reducer can differ in insertion/size-reduction scheduling;
    this trace is independently recorded and never silently treated as a
    native fplll callback. Native LLL determines all primary benchmark metrics.
    """
    from fpylll import LLL, IntegerMatrix
    from sandpile_mpfr import gso_mu_entry, gso_swap_is_valid
    B = IntegerMatrix(B0)
    M = _mat_gso(B, prec)
    L = LLL.Reduction(M, delta=float(delta), eta=float(eta))
    k = 1
    swaps = 0
    tests = 0
    valid_mu = []
    start = time.perf_counter()
    while k < B.nrows:
        # fpylll computes the Babai size reduction of row k, not Python
        # floating point rounding and not a standalone custom GSO routine.
        if not M.update_gso_row(k-1, k-1):
            raise ArithmeticError(
                f"Predecessor GSO update failed at k={k}"
            )
        L.size_reduction(k, k+1, 0)
        if not M.update_gso_row(k-1, k-1):
            raise ArithmeticError(f'GSO row {k-1} update failed')
        if not M.update_gso_row(k, k):
            raise ArithmeticError(f'GSO row {k} update failed')
        tests += 1
        valid = gso_swap_is_valid(M, k, delta, eta,
                                  siegel=(condition == 'siegel'))
        if valid:
            if swaps >= max_swaps:
                raise RuntimeError(f'trace exceeded max_swaps={max_swaps}')
            valid_mu.append(gso_mu_entry(M, k, k-1))
            # Adjacent permutation with GSO cache invalidation.
            # M.swap_rows() changes basis rows without invalidating cached
            # Gram--Schmidt coefficients.  Using it here can make the Lovasz
            # predicate repeatedly test stale values and hit max_swaps.
            # For adjacent rows, move_row(k, k-1) is the same permutation,
            # but fplll invalidates all affected GSO data correctly.
            M.move_row(k, k-1)
            swaps += 1
            k = max(1, k-1)
        else:
            k += 1
    if not M.update_gso():
        raise ArithmeticError('trace final GSO update failed')
    state = _snapshot(M, delta, eta)
    return {
        'source': 'instrumented_adjacent_replay_not_native_callback',
        'valid_swap_mu': valid_mu,
        'final_subdiagonal_mu': state['subdiag_mu'],
        'profile': state['r'],
        'swaps': swaps,
        'tests': tests,
        'runtime_sec': time.perf_counter()-start,
    }


def _simulated_lll(M_initial, condition, prec, delta, eta,
                   seed, max_swaps):
    from sandpile_mpfr import SandpileSeq
    s = SandpileSeq(M_initial, precision=prec, condition=condition,
                    delta=delta, eta=eta, seed=seed,
                    allow_lossy_gso=False)
    start = time.perf_counter()
    s.run(max_swaps=max_swaps)
    elapsed = time.perf_counter()-start
    state = s.snapshot()
    return {'profile': state['r'], 'mu': state['mu'],
            'valid_swap_mu': state['valid_swap_mu'],
            'final_subdiagonal_mu': state['final_subdiagonal_mu'],
            'metrics': metrics(state['r'], prec),
            'swaps': int(s.swaps), 'tests': int(s.tests),
            'runtime_sec': elapsed}


def _run_one(payload):
    root_str, spec, cfg = payload
    root = Path(root_str)
    instance_id = spec['id']
    out = root/'runs'/instance_id
    out.mkdir(parents=True, exist_ok=True)
    try:
        from fpylll import FPLLL
        prec = int(cfg.get('precision', 212))
        delta = float(cfg.get('delta', 0.99))
        eta = float(cfg.get('eta', 0.51))
        max_swaps = int(cfg.get('max_swaps', 1000000))
        n_sims = int(cfg.get('simulation_seeds_per_lattice', 3))
        if n_sims < 1:
            raise ValueError('simulation_seeds_per_lattice must be positive')
        FPLLL.set_threads(1)
        FPLLL.set_precision(prec)
        original = read_json_gz(root/spec['file'])
        B = _preprocess_size_reduce(from_rows(original['basis']), prec, delta, eta)
        write_json_gz(out/'size_reduced_input.json.gz', {
            'instance_id': instance_id, 'basis': matrix_to_rows(B),
            'delta': delta, 'eta': eta,
        })
        M_initial = _mat_gso(B, prec)
        initial = _snapshot(M_initial, delta, eta)
        write_json_gz(out/'initial_gso.json.gz', initial)
        output_paths = []
        for condition in ('lovasz', 'siegel'):
            native = _native_lll(B, condition, prec, delta, eta)
            native.update({'instance_id': instance_id, 'family': spec['family'],
                           'variant': spec['variant'], 'spec': spec['spec'],
                           'algorithm': condition, 'kind': 'real',
                           'delta': delta, 'eta': eta, 'precision': prec,
                           'status': 'finished'})
            if cfg.get('trace_real_swaps', True):
                try:
                    trace = _trace_before_swaps(B, condition, prec, delta,
                                                eta, max_swaps)
                    trace['profile_discrepancy'] = profile_distance(
                        native['profile'], trace['profile'], prec)
                    native['instrumented_trace'] = trace
                except Exception as exc:
                    native['instrumented_trace'] = {
                        'status': 'failed', 'error': repr(exc),
                        'traceback': traceback.format_exc(),
                    }
            native_file = out/f'{condition}_real.json.gz'
            write_json_gz(native_file, native)
            output_paths.append(str(native_file.relative_to(root)))
            for j in range(n_sims):
                # Stable seed independent of execution/scheduling order.
                sim_seed = (int(spec['seed']) * 1000003 + j*104729
                            + (1 if condition == 'lovasz' else 2)) % (2**31-1)
                try:
                    sim = _simulated_lll(M_initial, condition, prec, delta, eta,
                                         sim_seed, max_swaps)
                    sim['status'] = 'finished'
                    sim['errors'] = profile_distance(native['profile'],
                                                     sim['profile'], prec)
                except Exception as exc:
                    sim = {'status': 'failed', 'error': repr(exc),
                           'traceback': traceback.format_exc()}
                sim.update({'instance_id': instance_id,
                            'family': spec['family'], 'variant': spec['variant'],
                            'spec': spec['spec'], 'algorithm': condition,
                            'kind': 'sim', 'delta': delta, 'eta': eta,
                            'precision': prec, 'seed': sim_seed})
                sim_file = out/f'{condition}_sim_{j:03d}.json.gz'
                write_json_gz(sim_file, sim)
                output_paths.append(str(sim_file.relative_to(root)))
        return {'instance_id': instance_id, 'status': 'finished',
                'files': output_paths}
    except Exception as exc:
        error = {'instance_id': instance_id, 'status': 'failed',
                 'error': repr(exc), 'traceback': traceback.format_exc()}
        (out/'error.json').write_text(json.dumps(error, indent=2), encoding='utf-8')
        return error


def run_instances(root, workers=None, rerun=False):
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    config = manifest['config']
    jobs = []
    for inst in manifest['instances']:
        indir = root/'runs'/inst['id']
        n_seeds = int(config.get('simulation_seeds_per_lattice', 3))
        expected = [indir/f'{cond}_real.json.gz' for cond in ('lovasz','siegel')]
        expected += [indir/f'{cond}_sim_{j:03d}.json.gz'
                     for cond in ('lovasz','siegel') for j in range(n_seeds)]
        complete = all(p.exists() for p in expected)
        if rerun or not complete:
            jobs.append((str(root), inst, config))
    workers = int(workers or config.get('n_workers', 4))
    print(f'[run] {len(jobs)} lattice jobs; {workers} workers', flush=True)
    reports = []
    # Spawn is safer than fork for Sage/GMP/MPFR and independent RNGs.
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=mp.get_context('spawn')) as executor:
        fs = {executor.submit(_run_one, p): p[1]['id'] for p in jobs}
        for future in as_completed(fs):
            result = future.result()
            reports.append(result)
            print(f"[run] {result['instance_id']}: {result['status']}"
                  + (f" {result.get('error','')}" if result['status'] != 'finished' else ''),
                  flush=True)
    (root/'run_report.json').write_text(json.dumps(reports, indent=2),
                                        encoding='utf-8')
    return reports
