"""CLI: generate → run → summarize → plot (usable outside Jupyter)."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='Paired fpylll vs MPFR sandpile benchmark')
    parser.add_argument('command', choices=['generate','run','summarize','plot','all'])
    parser.add_argument('--config', default='example_config.json',
                        help='JSON config for generate/all')
    parser.add_argument('--output', default='benchmark_results')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true',
                        help='allow overwriting manifest on generation')
    parser.add_argument('--rerun',action='store_true',
                        help='rerun existing lattice jobs')
    args=parser.parse_args()
    out=Path(args.output)
    if args.command in ('generate','all'):
        from bench_generators import generate_instances
        config=json.loads(Path(args.config).read_text())
        generate_instances(config,out,overwrite=args.overwrite)
    if args.command in ('run','all'):
        from bench_runner import run_instances
        run_instances(out,workers=args.workers,rerun=args.rerun)
    if args.command in ('summarize','all'):
        from bench_stats import summarize
        summarize(out)
    if args.command in ('plot','all'):
        from bench_plot import plot_all
        plot_all(out)


if __name__=='__main__':
    main()
