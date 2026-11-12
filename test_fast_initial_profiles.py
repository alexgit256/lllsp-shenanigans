"""Quick tests: run in the Sage environment after building sandpile_mpfr."""
from mpmath import mp
from fast_initial_profiles import initial_qary, initial_intrel, initial_coppersmith


def test_qary():
    s = initial_qary(24, 9, 13.5, precision=212, seed=11)
    snap = s.snapshot()
    r, mu = snap['r'], snap['mu']
    with mp.workprec(200):
        for x in r[:9]:
            assert abs(mp.log(mp.mpf(x),2) - 27) < mp.mpf('1e-40')
        assert all(mp.mpf(x) == 1 for x in r[9:])
    assert all(mp.mpf(mu[i][j]) == 0 for i in range(9) for j in range(i))
    assert all(mp.mpf(mu[i][j]) == 0 for i in range(9,24) for j in range(9,i))


def test_intrel():
    s = initial_intrel(20, 1024, precision=212, seed=11)
    snap = s.snapshot()
    with mp.workprec(160):
        product = mp.mpf(1)
        for x in snap['r']:
            assert mp.mpf(x) > 0
            product *= mp.mpf(x)
        assert product > 1
    assert any(abs(float(snap['mu'][i][j])) > 0.01 for i in range(1,20) for j in range(i))


def test_coppersmith():
    d,m,t,X = 2,4,2,8
    s = initial_coppersmith(d,m,t,32,X,precision=212,seed=11)
    snap = s.snapshot()
    assert len(snap['r']) == d*m+t
    assert all(mp.mpf(x)>0 for x in snap['r'])
    assert any(mp.mpf(snap['mu'][i][j]) != 0 for i in range(1,d*m+t) for j in range(i))



# ---------------------------------------------------------------------------
# Non-blocked LLL-SP benchmark.  Families are configured in a DICTIONARY.
# The NTRU samples are circulant q-ary proxies, NOT genuine f,g NTRU.
# ---------------------------------------------------------------------------

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

# Extend this dictionary with other families/factories as required.  Each family
# is a set of 20 independent lattice-profile seeds, crossed with conditions and
# precisions.  Factory names resolve in PROFILE_FACTORIES below.
FAMILIES = {
    f"ntru_{n}": {
        "factory": "ntru_circulant_proxy",
        "dimension": n,
        "q_bits": 14.2 * n / 128,   # uncalibrated heuristic
        "count": 20,
        # (fptype, precision): precision is meaningful ONLY for MPFR.
        # Native double is represented with precision=None.
        "backends": (("mpfr", 103), ("mpfr", 206), ("mpfr", 412),
                     ("double", None)) if n in (128, 256)
                    else (("double", None),),
        "conditions": ("lovasz", "siegel"),
    }
    for n in (128, 256, 512, 768, 1024)
}

RAW_FIELDS = ("case_id", "family", "dimension", "instance", "seed", "condition",
              "fptype", "precision", "q_bits", "q", "delta", "eta", "max_swaps",
              "generation_seconds", "simulation_seconds", "total_seconds",
              "swaps", "tests", "swap_limit_hit", "error")
SUMMARY_FIELDS = ("family", "dimension", "condition", "fptype", "precision", "n_ok",
                  "n_error", "n_limit_hit", "runtime_median_s", "runtime_variance_s2",
                  "swaps_median", "swaps_variance", "generation_median_s",
                  "total_walltime_s", "simulation_walltime_s")


def ntru_circulant_proxy(case, *, delta, eta):
    """Exact GSO of [qI,0]; [circulant(h),I], NOT of planted f/g NTRU.

    Structure: r=[q^2]*(n/2)+[1]*(n/2), lower-left mu=circulant(h)/q.
    Samples one independent h per instance, reused across conditions and p.
    """
    from mpmath import mp
    from sandpile_mpfr import SandpileSeq

    n = case["dimension"]
    d = n // 2
    if n % 2:
        raise ValueError("NTRU proxy requires an even ambient dimension")
    prec = case["precision"] if case["fptype"] == "mpfr" else 53
    with mp.workprec(max(128, prec + 48)):
        q = int(mp.ceil(mp.power(2, mp.mpf(str(case["q_bits"])))))
        digits = int(prec * 0.30103) + 12
        rng = random.Random(case["seed"])
        h = [rng.randrange(q) for _ in range(d)]
        h = [v if 2*v <= q else v-q for v in h]
        ratios = [mp.nstr(mp.mpf(v)/q, n=digits) for v in h]
    mu = [["0"]*n for _ in range(n)]
    for i in range(d):
        row = mu[d+i]
        for j in range(d):
            row[j] = ratios[(j-i) % d]
    sim = SandpileSeq((([str(q*q)]*d)+(["1"]*d), mu),
                     condition=case["condition"], delta=delta, eta=eta,
                     precision=case["precision"], fptype=case["fptype"],
                     seed=case["seed"], trace_swaps=False)
    return sim, q


PROFILE_FACTORIES = {"ntru_circulant_proxy": ntru_circulant_proxy}


def cases_for_families(families=FAMILIES, *, base_seed=1729,
                       count_override=None, selected=None, delta=0.99,
                       eta=0.51, max_swaps=1_000_000):
    """Return a deterministic experiment inventory from the family dict."""
    names = list(selected) if selected else list(families)
    out = []
    for name in names:
        cfg = families[name]
        if cfg["factory"] not in PROFILE_FACTORIES:
            raise ValueError(f"unknown factory for {name}: {cfg['factory']}")
        count = cfg["count"] if count_override is None else count_override
        if count < 1:
            raise ValueError("count must be positive")
        for j in range(count):
            # Stable across condition and precision. Do not use Python hash().
            seed = int.from_bytes(hashlib.sha256(
                f"{base_seed}/{name}/{j}".encode()).digest()[:4], 'big')
            for condition in cfg["conditions"]:
                for fptype, prec in cfg["backends"]:
                    base = dict(family=name, factory=cfg["factory"],
                                dimension=cfg["dimension"], instance=j,
                                seed=seed, condition=condition,
                                fptype=fptype, precision=prec,
                                q_bits=cfg["q_bits"], delta=delta, eta=eta,
                                max_swaps=max_swaps)
                    # A content-addressed ID prevents accidental reuse after
                    # parameter changes and supports append/resume.
                    ident = hashlib.sha256(json.dumps(base, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()[:20]
                    base["case_id"] = ident
                    out.append(base)
    return out


def _read_rows(path):
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(RAW_FIELDS):
            raise ValueError(f"CSV schema mismatch in {path}; choose a new output path")
        return list(reader)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def _stat(values, fn):
    return fn(values) if values else float('nan')


def summarize(rows, cases):
    """Per-family + condition + backend + precision stats over lattice seeds."""
    ids = {c["case_id"] for c in cases}
    by_key = defaultdict(dict)
    for r in rows:
        if r.get("case_id") in ids:
            key = (r["family"], int(r["dimension"]), r["condition"],
                   r["fptype"], int(r["precision"]) if r["precision"] else None)
            by_key[key][r["case_id"]] = r  # latest attempt wins
    inventory = defaultdict(list)
    for c in cases:
        inventory[(c["family"], c["dimension"], c["condition"],
                   c["fptype"], c["precision"])].append(c)
    result = []
    for key, listed in inventory.items():
        attempts = by_key.get(key, {})
        successful = [r for r in attempts.values() if not r["error"]]
        times = [_num(r["simulation_seconds"]) for r in successful]
        swaps = [int(r["swaps"]) for r in successful]
        gens = [_num(r["generation_seconds"]) for r in successful]
        def median(x): return statistics.median(x) if x else float('nan')
        def var(x): return statistics.variance(x) if len(x)>1 else (0.0 if x else float('nan'))
        result.append(dict(zip(SUMMARY_FIELDS, (
            key[0], key[1], key[2], key[3], key[4], len(successful),
            len(attempts)-len(successful),
            sum(int(r["swap_limit_hit"]) for r in successful),
            median(times), var(times), median(swaps), var(swaps),
            median(gens),
            sum(_num(r["total_seconds"]) for r in attempts.values()),
            sum(_num(r["simulation_seconds"]) for r in attempts.values()
                if r["simulation_seconds"]),
        ))))
    return sorted(result, key=lambda x: (x["dimension"], x["condition"],
                                               x["fptype"], x["precision"] or 0))


def write_summary(path, groups):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(groups)


def print_summary(groups, *, wall_seconds=None):
    print("\nPer-family simulation statistics (sample variance, n-1):")
    print(f"{'Family':<12} {'Condition':<8} {'type':<7} {'p':>4} {'ok':>4} {'err':>3} "
          f"{'limit':>5} {'med s':>12} {'var s^2':>14} {'med swaps':>14} {'var swaps':>17} {'wall s':>12}")
    for g in groups:
        def fmt(v):
            return f"{v:.5g}" if isinstance(v, (float,int)) and math.isfinite(v) else 'n/a'
        print(f"{g['family']:<12} {g['condition']:<8} {g['fptype']:<7} "
              f"{str(g['precision'] or '-'):>4} "
              f"{g['n_ok']:>4} {g['n_error']:>3} {g['n_limit_hit']:>5} "
              f"{fmt(g['runtime_median_s']):>12} {fmt(g['runtime_variance_s2']):>14} "
              f"{fmt(g['swaps_median']):>14} {fmt(g['swaps_variance']):>17} "
              f"{fmt(g['total_walltime_s']):>12}")
    if wall_seconds is not None:
        print(f"Current benchmark invocation wall time: {wall_seconds:.3f} s")
    print("Group wall s = sum of measured per-case generation+simulation times; "
          "not necessarily total wall-clock time (e.g. resumed runs).")


def run_benchmarks(cases, *, output, resume=True, report=None):
    """Execute both conditions for every configured backend, recording all runs."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_rows(output) if resume else []
    done = {r['case_id'] for r in existing if not r['error']}
    if not resume and output.exists():
        output.unlink()
    t_campaign = time.perf_counter()
    with output.open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        if f.tell() == 0:
            writer.writeheader()
            f.flush()
        for ix, c in enumerate(cases, 1):
            if c['case_id'] in done:
                continue
            r = {name: c.get(name, '') for name in RAW_FIELDS}
            r.update(q='', generation_seconds='', simulation_seconds='',
                     total_seconds='', swaps='', tests='', swap_limit_hit='', error='')
            t0 = time.perf_counter()
            print(f"[{ix}/{len(cases)}] {c['family']} #{c['instance']:02d} "
                  f"{c['condition']} {c['fptype']} p={c['precision'] or '-'}", flush=True)
            try:
                sim, q = PROFILE_FACTORIES[c['factory']](
                    c, delta=c['delta'], eta=c['eta'])
                r['q'] = str(q)
                r['generation_seconds'] = f"{time.perf_counter()-t0:.9f}"
                ts = time.perf_counter()
                try:
                    sim.run(max_swaps=c['max_swaps'])  # non-blocked
                finally:
                    r['simulation_seconds'] = f"{time.perf_counter()-ts:.9f}"
                    r['swaps'] = int(sim.swaps)
                    r['tests'] = getattr(sim, 'tests', '')
                    r['swap_limit_hit'] = int(sim.swaps >= c['max_swaps'])
            except Exception as exc:
                r['error'] = f"{type(exc).__name__}: {exc}"
            finally:
                r['total_seconds'] = f"{time.perf_counter()-t0:.9f}"
                writer.writerow(r)
                f.flush()
            if r['error']:
                print(f"  ERROR {r['error']}", flush=True)
            else:
                print(f"  sim {r['simulation_seconds']} s; "
                      f"swaps={r['swaps']}; total={r['total_seconds']} s", flush=True)
    rows = _read_rows(output)
    groups = summarize(rows, cases)
    write_summary(Path(report) if report else output.with_name(output.stem+'_summary.csv'), groups)
    print_summary(groups, wall_seconds=time.perf_counter()-t_campaign)
    return groups


def cli(argv=None):
    parser = argparse.ArgumentParser(description="LLL-SP Lovasz/Siegel benchmark (MPFR/double)")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--bench', action='store_true', help='execute simulations')
    actions.add_argument('--list', action='store_true', help='print JSON experiment inventory')
    actions.add_argument('--report', action='store_true', help='summarize existing CSV')
    parser.add_argument('--families', nargs='+', choices=tuple(FAMILIES))
    parser.add_argument('--count', type=int, default=None,
                        help='override instances per family (default 20)')
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--max-swaps', type=lambda x: int(x, 0), default=1_000_000,
                        help="swap limit, decimal or e.g. 0x400000000")
    parser.add_argument('--delta', type=float, default=0.99)
    parser.add_argument('--eta', type=float, default=0.51)
    parser.add_argument('--output', default='ntru_sp_benchmark.csv')
    parser.add_argument('--summary', default=None)
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args(argv)
    if args.max_swaps < 1 or (args.count is not None and args.count < 1):
        parser.error('count and max-swaps must be positive')
    cases = cases_for_families(base_seed=args.seed, count_override=args.count,
                               selected=args.families, delta=args.delta,
                               eta=args.eta, max_swaps=args.max_swaps)
    if args.list:
        print(json.dumps(cases, indent=2))
    elif args.bench:
        run_benchmarks(cases, output=args.output, resume=not args.no_resume,
                       report=args.summary)
    elif args.report:
        rows = _read_rows(Path(args.output))
        result = summarize(rows, cases)
        write_summary(Path(args.summary) if args.summary else
                      Path(args.output).with_name(Path(args.output).stem+'_summary.csv'),
                      result)
        print_summary(result)
    else:
        test_qary()
        test_intrel()
        test_coppersmith()
        print('All fast-profile smoke tests passed')


if __name__ == '__main__':
    cli()
