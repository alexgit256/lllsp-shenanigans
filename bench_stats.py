"""Separate-family paired statistics; can run without Sage or fpylll."""
from __future__ import annotations
import csv
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path
from bench_generators import read_json_gz, write_json_gz

METRICS = ('rhf', 'log2_rhf', 'slope', 'log2_potential',
           'swaps', 'runtime_sec')
ERRORS = ('profile_mae', 'profile_rmse', 'profile_max_abs', 'profile_bias')


def _mean(values):
    return stats.fmean(values) if values else float('nan')


def _sd(values):
    return stats.stdev(values) if len(values)>1 else (0.0 if values else float('nan'))


def _quantile(values, q):
    if not values:
        return float('nan')
    v=sorted(values)
    k=(len(v)-1)*q
    j=int(k)
    if j+1 == len(v):
        return v[j]
    return v[j] + (k-j)*(v[j+1]-v[j])


def description(values):
    v = [float(x) for x in values if math.isfinite(float(x))]
    if not v:
        return {'n': 0}
    m = _mean(v)
    sd = _sd(v)
    return {'n':len(v), 'mean':m, 'std':sd, 'median':stats.median(v),
            'q05':_quantile(v, .05), 'q25':_quantile(v, .25),
            'q75':_quantile(v, .75), 'q95':_quantile(v, .95),
            'mean_ci95_normal': [m-1.96*sd/math.sqrt(len(v)),
                                 m+1.96*sd/math.sqrt(len(v))]}


def _val(record, name):
    return float(record['metrics'][name] if name in record['metrics'] else record[name])


def _log_profile(record):
    # Convert to binary64 ONLY for group plotting; canonical profiles remain
    # high precision decimal strings in individual run files.
    from decimal import Decimal, localcontext
    with localcontext() as ctx:
        ctx.prec = 40
        ln2 = Decimal(2).ln()
        return [float(Decimal(str(x)).ln() / ln2) for x in record['profile']]


def _profile_aggregate(profile_series):
    if not profile_series:
        return {}
    n = len(profile_series[0])
    if any(len(row)!=n for row in profile_series):
        raise ValueError('mixed profile dimensions in one statistic group')
    return {'n':len(profile_series),
            'mean':[_mean([v[i] for v in profile_series]) for i in range(n)],
            'std':[_sd([v[i] for v in profile_series]) for i in range(n)]}


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def _key(item):
    return (item['family'], int(item['variant']),
            len(item['profile']), item['algorithm'])


def summarize(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    results=defaultdict(lambda: {'real':None, 'sim':[]})
    failures=[]
    for instance in manifest['instances']:
        idir=root/'runs'/instance['id']
        for cond in ('lovasz','siegel'):
            fname=idir/f'{cond}_real.json.gz'
            if not fname.exists():
                failures.append({'instance':instance['id'],'condition':cond,
                                 'reason':'real output missing'})
                continue
            real=read_json_gz(fname)
            if real.get('status')!='finished':
                failures.append({'instance':instance['id'],'condition':cond,
                                 'reason':'real output failed'})
                continue
            key=_key(real)
            pair=results[(key,instance['id'])]
            pair['real']=real
            for simfile in sorted(idir.glob(f'{cond}_sim_*.json.gz')):
                sim=read_json_gz(simfile)
                if sim.get('status')=='finished':
                    pair['sim'].append(sim)
                else:
                    failures.append({'instance':instance['id'],'condition':cond,
                                     'reason':sim.get('error', 'sim failed')})
    grouped=defaultdict(list)
    for (key, instance_id), pair in results.items():
        if pair['real'] is not None and pair['sim']:
            grouped[key].append((instance_id, pair['real'], pair['sim']))
    groups_output=[]
    profile_output={}
    pair_rows=[]
    stat_rows=[]
    for key, paired in sorted(grouped.items()):
        family, variant, dimension, condition=key
        group_id=f'{family}_v{variant:02d}_n{dimension}_{condition}'
        scalars={}
        for metric in METRICS:
            real_vals=[_val(real, metric) for _,real,_ in paired]
            sim_vals=[_mean([_val(s,metric) for s in sims])
                      for _,_,sims in paired]
            diffs=[s-r for s,r in zip(sim_vals,real_vals)]
            scalars[metric]={
                'real':description(real_vals),
                'sim_lattice_means':description(sim_vals),
                'paired_error_sim_minus_real':description(diffs),
                'paired_rmse':math.sqrt(_mean([d*d for d in diffs])),
            }
            stat_rows.append({'group':group_id,'family':family,'variant':variant,
                              'dimension':dimension,'condition':condition,
                              'metric':metric,
                              'real_mean':_mean(real_vals),'real_std':_sd(real_vals),
                              'sim_mean':_mean(sim_vals),'sim_std':_sd(sim_vals),
                              'paired_bias':_mean(diffs),
                              'paired_rmse':scalars[metric]['paired_rmse'],
                              'n_lattices':len(paired)})
        for error in ERRORS:
            # Summaries of each lattice's *mean* simulation error.
            vals=[_mean([float(s['errors'][error]) for s in sims])
                  for _,_,sims in paired]
            scalars[error]=description(vals)
            stat_rows.append({'group':group_id,'family':family,'variant':variant,
                              'dimension':dimension,'condition':condition,
                              'metric':error,'sim_mean':_mean(vals),
                              'sim_std':_sd(vals),'n_lattices':len(paired)})
        raw_real_profiles=[_log_profile(real) for _,real,_ in paired]
        mean_sim_profiles=[]
        for _,_,sims in paired:
            profs=[_log_profile(s) for s in sims]
            mean_sim_profiles.append([_mean([p[j] for p in profs])
                                      for j in range(dimension)])
        profile_output[group_id]={
            'family':family,'variant':variant,'dimension':dimension,
            'condition':condition,
            'real':_profile_aggregate(raw_real_profiles),
            'sim_lattice_means':_profile_aggregate(mean_sim_profiles),
        }
        count_trace_ok = sum('valid_swap_mu' in real.get('instrumented_trace',{})
                             for _,real,_ in paired)
        groups_output.append({'id':group_id, 'family':family,
                              'variant':variant,'dimension':dimension,
                              'condition':condition, 'n_lattices':len(paired),
                              'n_traces_available':count_trace_ok,
                              'statistics':scalars})
        for instance_id, real, sims in paired:
            row={'instance_id':instance_id,'group':group_id,
                 'sim_repeats':len(sims),
                 'trace_swaps':real.get('instrumented_trace',{}).get('swaps'),
                 'native_swaps':real['swaps']}
            for metric in METRICS:
                rv=_val(real,metric)
                sv=_mean([_val(s,metric) for s in sims])
                row[f'real_{metric}']=rv
                row[f'sim_mean_{metric}']=sv
                row[f'paired_err_{metric}']=sv-rv
            for error in ERRORS:
                row[f'mean_{error}']=_mean([float(s['errors'][error]) for s in sims])
            pair_rows.append(row)
    output={'schema':1, 'n_groups':len(groups_output),
            'group_definitions':'grouped by family, variant, dimension and condition',
            'sampling_unit':'lattice; simulation repeats averaged per lattice',
            'groups':groups_output, 'failures':failures,
            'notes':['Plot arrays are binary64 summaries; raw run profiles are MPFR strings.',
                     'Real valid-swap mu samples come from a separately instrumented adjacent-swap replay.',
                     'Bootstrap or numerical invariant validation is postponed.']}
    (root/'aggregate.json').write_text(json.dumps(output, indent=2),encoding='utf-8')
    write_json_gz(root/'profiles_summary.json.gz',profile_output)
    _write_csv(root/'summary.csv',stat_rows)
    _write_csv(root/'paired.csv',pair_rows)
    print(f'[summarize] {len(groups_output)} groups, {len(pair_rows)} paired lattices, {len(failures)} issues')
    return output
