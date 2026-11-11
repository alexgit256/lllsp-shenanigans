"""Headless, per-family diagnostics. Reads completed on-disk results only."""
from __future__ import annotations
import math
from pathlib import Path
from bench_generators import read_json_gz


def plot_all(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    root=Path(root)
    plots=root/'plots'
    plots.mkdir(exist_ok=True, parents=True)
    groups=read_json_gz(root/'profiles_summary.json.gz')
    names=[]
    for group, info in sorted(groups.items()):
        real=info['real']; sim=info['sim_lattice_means']
        x=np.arange(info['dimension'])
        fig, ax=plt.subplots(figsize=(9,5))
        for label,data in [('fpylll',real),('SandpileSeq',sim)]:
            avg=np.asarray(data['mean'], dtype=float)
            std=np.asarray(data['std'], dtype=float)
            line=ax.plot(x,avg,label=label)[0]
            ax.fill_between(x, avg-std, avg+std, alpha=.17,
                            color=line.get_color())
        ax.set(title=f'{group}: mean ± SD across lattices',
               xlabel='GSO index i', ylabel='log₂ rᵢ (squared GS length)')
        ax.legend()
        fig.tight_layout()
        fname=plots/f'{group}_profiles.png'
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        names.append(fname)
    import json
    aggregate=json.loads((root/'aggregate.json').read_text())
    for group in aggregate['groups']:
        gid=group['id']; family=group['family']; cond=group['condition']
        variant=group['variant']; dimension=group['dimension']
        # Match simulator records to the exact same lattice IDs that have
        # successful real traces. Otherwise pooled distributions are not paired.
        import warnings
        n_lattices=0
        n_traced=0
        missing=[]
        real_valid_count=sim_valid_count=0
        real_final_count=sim_final_count=0
        eta=float(aggregate.get('config', {}).get('eta', .51))
        bins=np.linspace(-max(.55,eta+.04),max(.55,eta+.04),51)
        real_valid_hist=np.zeros(len(bins)-1,dtype=np.int64)
        sim_valid_hist=np.zeros_like(real_valid_hist)
        real_final_hist=np.zeros_like(real_valid_hist)
        sim_final_hist=np.zeros_like(real_valid_hist)
        for idir in sorted((root/'runs').iterdir()):
            if not idir.is_dir():
                continue
            real_file=idir/f'{cond}_real.json.gz'
            if not real_file.exists():
                continue
            rec=read_json_gz(real_file)
            if rec.get('status') != 'finished':
                continue
            if (rec['family'],rec['variant'],len(rec['profile'])) != (family,variant,dimension):
                continue
            n_lattices += 1
            trace=rec.get('instrumented_trace',{})
            real_mu=trace.get('valid_swap_mu',[])
            if not real_mu:
                missing.append(f"{idir.name}: {trace.get('error','missing or empty valid_swap_mu')}")
            else:
                n_traced += 1
            def accumulate(hist, values):
                if not values:
                    return 0
                a=np.fromiter((float(v) for v in values), dtype=float, count=len(values))
                if not np.isfinite(a).all():
                    raise ValueError('nonfinite mu values')
                hist += np.histogram(a,bins=bins)[0]
                return len(a)
            # Terminal μ distribution: native + every corresponding simulation.
            real_final_count += accumulate(real_final_hist,rec.get('final_subdiagonal_mu',[]))
            for fp in sorted(idir.glob(f'{cond}_sim_*.json.gz')):
                sim=read_json_gz(fp)
                if sim.get('status') != 'finished':
                    continue
                sim_final_count += accumulate(sim_final_hist,sim.get('final_subdiagonal_mu',[]))
                # Only compare valid-swap events for lattices with BOTH sources.
                if real_mu:
                    sim_valid_count += accumulate(sim_valid_hist,sim.get('valid_swap_mu',[]))
            if real_mu:
                real_valid_count += accumulate(real_valid_hist,real_mu)
        if missing:
            warnings.warn(f'{gid}: {len(missing)} of {n_lattices} real traces missing; '
                          f'first issue: {missing[0]}. '
                          f'Run repair_real_traces.py before drawing conclusions.',
                          RuntimeWarning)
            print(f'[trace WARNING] {gid}: missing {len(missing)}/{n_lattices}: '
                  f'{missing[0]}',flush=True)
        def draw_hist(ax,hist,num,label):
            if num == 0 or hist.sum() == 0:
                return
            density=hist/(hist.sum()*np.diff(bins))
            ax.stairs(density,bins,label=f'{label} (n={num:,})',linewidth=1.5)
        # Pre-valid-swap: NEVER substitute a simulated-only chart as comparison.
        fig,ax=plt.subplots(figsize=(8,4.5))
        if real_valid_count and sim_valid_count:
            draw_hist(ax,real_valid_hist,real_valid_count,'Real LLL (instrumented replay)')
            draw_hist(ax,sim_valid_hist,sim_valid_count,'Simulated')
            subtitle=f'{n_traced}/{n_lattices} lattices with a successful real trace'
            if missing:
                subtitle += ' — INCOMPLETE'
        else:
            subtitle='NO REAL/SIM PAIRED PRE-SWAP DATA'
            ax.text(.5,.5,'Real LLL trace missing or empty\nRun repair_real_traces.py',
                    transform=ax.transAxes, ha='center',va='center',fontsize=13,
                    color='darkred')
        ax.set(xlabel='μᵢ₊₁,ᵢ',ylabel='Event-weighted density',
               title=f'{gid}: before valid swap\n{subtitle}')
        if real_valid_count and sim_valid_count:
            ax.legend()
        fig.tight_layout()
        fname=plots/f'{gid}_before_valid_swap.png'
        fig.savefig(fname,dpi=150)
        plt.close(fig)
        names.append(fname)
        # Terminal μ: native LLL's final GSO and simulator's final GSO.
        fig,ax=plt.subplots(figsize=(8,4.5))
        draw_hist(ax,real_final_hist,real_final_count,'Native real LLL')
        draw_hist(ax,sim_final_hist,sim_final_count,'Simulated')
        ax.set(xlabel='μᵢ₊₁,ᵢ',ylabel='Coefficient density',
               title=f'{gid}: final μ')
        if real_final_count or sim_final_count:
            ax.legend()
        fig.tight_layout()
        fname=plots/f'{gid}_final_mu.png'
        fig.savefig(fname,dpi=150)
        plt.close(fig)
        names.append(fname)
    # Paired per-lattice metric comparison; avoids inflating n for repeated seeds.
    import csv
    with open(root/'paired.csv',newline='',encoding='utf-8') as fd:
        paired=list(csv.DictReader(fd))
    for group in sorted(groups):
        group_rows=[r for r in paired if r['group']==group]
        for metric in ('rhf','slope','log2_potential','swaps'):
            xs=[float(r[f'real_{metric}']) for r in group_rows]
            ys=[float(r[f'sim_mean_{metric}']) for r in group_rows]
            if not xs:
                continue
            lo=min(xs+ys); hi=max(xs+ys)
            padding=.05*(hi-lo) if hi>lo else max(.001,abs(lo)*.05)
            fig,ax=plt.subplots(figsize=(6,5))
            ax.scatter(xs,ys,alpha=.8)
            ax.plot([lo-padding,hi+padding],[lo-padding,hi+padding],
                    linestyle='--',linewidth=1,alpha=.6)
            ax.set(xlim=(lo-padding,hi+padding),ylim=(lo-padding,hi+padding),
                   xlabel=f'Native fpylll: {metric}',
                   ylabel=f'Simulated lattice-mean: {metric}',
                   title=f'{group}: paired {metric} (n={len(xs)})')
            fig.tight_layout()
            fname=plots/f'{group}_paired_{metric}.png'
            fig.savefig(fname,dpi=150)
            plt.close(fig)
            names.append(fname)
    print(f'[plot] {len(names)} plots in {plots}')
    return names
