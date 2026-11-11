"""Repair failed/missing real pre-swap mu traces without rerunning LLL or simulations.

Run this in the benchmark source directory with its compiled sandpile_mpfr.so
and the existing results directory. Only updates the `instrumented_trace`
field of completed *_real.json.gz records after successful replays.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from bench_generators import from_rows, read_json_gz, write_json_gz
from bench_runner import _trace_before_swaps, profile_distance


def _selected(instance, family):
    return family is None or instance.get('family') == family


def run(root, *, family=None, conditions=('lovasz', 'siegel'),
        force=False, limit=None, max_swaps_override=None, dry_run=False):
    from fpylll import FPLLL
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    cfg = manifest['config']
    precision = int(cfg.get('precision', 212))
    delta = float(cfg.get('delta', 0.99))
    eta = float(cfg.get('eta', 0.51))
    max_swaps = (int(max_swaps_override) if max_swaps_override is not None
                 else int(cfg.get('max_swaps', 1000000)))
    FPLLL.set_threads(1)
    FPLLL.set_precision(precision)
    seen = ok = failed = skipped = 0
    for instance in manifest['instances']:
        if not _selected(instance, family):
            continue
        idir = root / 'runs' / instance['id']
        input_file = idir / 'size_reduced_input.json.gz'
        for condition in conditions:
            real_file = idir / f'{condition}_real.json.gz'
            if not real_file.exists():
                print(f'[skip] {instance["id"]}/{condition}: no native result')
                skipped += 1
                continue
            native = read_json_gz(real_file)
            if native.get('status') != 'finished':
                print(f'[skip] {instance["id"]}/{condition}: native run did not finish')
                skipped += 1
                continue
            previous = native.get('instrumented_trace') or {}
            if not force and previous.get('valid_swap_mu'):
                print(f'[skip] {instance["id"]}/{condition}: existing trace: '
                      f'{len(previous["valid_swap_mu"])} events')
                skipped += 1
                continue
            if limit is not None and seen >= limit:
                break
            seen += 1
            reason = previous.get('error', 'missing/empty trace')
            print(f'[trace] {instance["id"]}/{condition}: retrying; previous={reason}', flush=True)
            if not input_file.exists():
                print(f'[error] missing saved size-reduced basis: {input_file}', flush=True)
                failed += 1
                continue
            if dry_run:
                continue
            try:
                B = from_rows(read_json_gz(input_file)['basis'])
                start = time.perf_counter()
                trace = _trace_before_swaps(B, condition, precision,
                                            delta, eta, max_swaps)
                if not trace.get('valid_swap_mu'):
                    raise RuntimeError('trace completed without swaps; check native swaps')
                trace['profile_discrepancy'] = profile_distance(
                    native['profile'], trace['profile'], precision)
                trace['status'] = 'finished'
                native['instrumented_trace'] = trace
                tmp = real_file.with_name(real_file.name + '.tmp')
                try:
                    write_json_gz(tmp, native)
                    os.replace(tmp, real_file)
                finally:
                    tmp.unlink(missing_ok=True)
                print(f'[ok] {instance["id"]}/{condition}: '
                      f'{len(trace["valid_swap_mu"])} mu events, '
                      f'{trace["swaps"]} replay swaps, '
                      f'{time.perf_counter()-start:.1f}s', flush=True)
                ok += 1
            except Exception as exc:
                # Preserve original real result; print full diagnostic immediately.
                import traceback
                print(f'[FAILED] {instance["id"]}/{condition}: {exc!r}',
                      file=sys.stderr, flush=True)
                traceback.print_exc()
                failed += 1
        if limit is not None and seen >= limit:
            break
    print(f'[trace] attempted={seen}, repaired={ok}, failed={failed}, skipped={skipped}')
    return failed == 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='results', help='Existing benchmark results directory')
    p.add_argument('--family', default=None, help='Filter: ntru, qary, intrel, coppersmith')
    p.add_argument('--condition', choices=['lovasz','siegel','both'], default='both')
    p.add_argument('--force', action='store_true', help='Replace successful traces as well')
    p.add_argument('--limit', type=int, default=None, help='Maximum trace attempts')
    p.add_argument('--max-swaps', type=int, default=None, help='Override saved max_swaps')
    p.add_argument('--dry-run', action='store_true')
    a=p.parse_args()
    conds=('lovasz','siegel') if a.condition=='both' else (a.condition,)
    if not run(a.output,family=a.family,conditions=conds,
               force=a.force,limit=a.limit,
               max_swaps_override=a.max_swaps, dry_run=a.dry_run):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
