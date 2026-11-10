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
        real_valid=[]; sim_valid=[]; real_final=[]; sim_final=[]
        for idir in (root/'runs').iterdir():
            if not idir.is_dir():
                continue
            real_file=idir/f'{cond}_real.json.gz'
            if not real_file.exists():
                continue
            rec=read_json_gz(real_file)
            if (rec['family'],rec['variant'],len(rec['profile'])) != (family,variant,dimension):
                continue
            trace=rec.get('instrumented_trace',{})
            real_valid.extend(float(x) for x in trace.get('valid_swap_mu',[]))
            real_final.extend(float(x) for x in rec['final_subdiagonal_mu'])
            for fp in sorted(idir.glob(f'{cond}_sim_*.json.gz')):
                sim=read_json_gz(fp)
                if sim.get('status') != 'finished':
                    continue
                sim_valid.extend(float(x) for x in sim.get('valid_swap_mu',[]))
                sim_final.extend(float(x) for x in sim['final_subdiagonal_mu'])
        bins=np.linspace(-.55,.55,51)
        for suffix, series in [('before_valid_swap',[(real_valid,'Instrumented real LLL'),
                                                      (sim_valid,'Simulated')]),
                               ('final_mu',[(real_final,'Native real LLL'),
                                            (sim_final,'Simulated')])]:
            fig,ax=plt.subplots(figsize=(8,4.5))
            for data,label in series:
                if data:
                    ax.hist(data,bins=bins,histtype='step',density=True,
                            label=f'{label} (n={len(data)})')
            ax.set(xlabel='μᵢ₊₁,ᵢ',ylabel='Event-weighted density',
                   title=f'{gid}: {suffix.replace("_", " ")}')
            ax.legend()
            fig.tight_layout()
            fname=plots/f'{gid}_{suffix}.png'
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
