"""CLI for reproducible, resume-safe block LLL paired experiments."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from block_bench_runner import generate, run, list_jobs
from block_bench_core import completed_run, read_record


def main(argv=None):
    p = argparse.ArgumentParser(description='Paired real vs simulated scheduled block LLL')
    p.add_argument('command', choices=('generate','run','all','status'))
    p.add_argument('--config', default='block_bench_smoke.json')
    p.add_argument('--output', default='results/block_bench')
    p.add_argument('--force', action='store_true', help='rerun all jobs (never regenerates cached instances)')
    p.add_argument('--max-jobs', type=int, default=None, help='limit new job attempts for a quick trial')
    args = p.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.command in ('generate','all'):
        generate(cfg, args.output)
    if args.command in ('run','all'):
        result = run(cfg, args.output, force=args.force, max_jobs=args.max_jobs)
        print(json.dumps(result, indent=2))
        if result['failed']:
            return 1
    if args.command == 'status':
        status = {'finished':0, 'failed':0, 'missing':0}
        for inst, job, path in list_jobs(cfg, args.output):
            if completed_run(path, job['job_key']):
                status['finished'] += 1
            elif path.exists():
                status['failed'] += 1
            else:
                status['missing'] += 1
        print(json.dumps(status, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
