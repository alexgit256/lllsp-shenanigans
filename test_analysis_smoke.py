"""Tests summary/plot pipeline with fake saved run data (no fpylll required)."""
from tempfile import TemporaryDirectory
import json
from pathlib import Path
from bench_generators import write_json_gz
from bench_stats import summarize
from bench_plot import plot_all


def _metrics(profile, swaps, runtime):
    return {'rhf':'1.01','log2_rhf':'0.015','slope':'-0.05',
            'log2_potential':'12','log2_first_r':'2',
            'log2_covolume':'2.2','dimension':2}


def main():
    with TemporaryDirectory() as td:
        root=Path(td)
        root.joinpath('manifest.json').write_text(json.dumps({
            'config':{},'instances':[{'id':'qary_v00_00000', 'file':'unused',
                                      'family':'qary','variant':0}],
        }))
        d=root/'runs'/'qary_v00_00000'
        d.mkdir(parents=True)
        for condition in ('lovasz','siegel'):
            fake_real={'instance_id':'qary_v00_00000','family':'qary','variant':0,
                       'algorithm':condition,'kind':'real','status':'finished',
                       'profile':['4','2'],'metrics':_metrics(['4','2'],2,0.01),
                       'swaps':2,'runtime_sec':0.01,
                       'final_subdiagonal_mu':['0.1'],
                       'instrumented_trace':{'valid_swap_mu':['0.2'],
                                             'swaps':2},
                       'mu':[['0','0'],['0.1','0']]}
            fake_sim={**fake_real,'kind':'sim','profile':['3','2.5'],
                      'valid_swap_mu':['0.25'],
                      'swaps':3,'runtime_sec':0.001,
                      'errors':{'profile_mae':'0.45','profile_rmse':'0.5',
                                'profile_max_abs':'0.6','profile_bias':'0.01'}}
            write_json_gz(d/f'{condition}_real.json.gz',fake_real)
            write_json_gz(d/f'{condition}_sim_000.json.gz',fake_sim)
        summary=summarize(root)
        assert summary['n_groups']==2
        assert (root/'summary.csv').exists()
        assert (root/'paired.csv').exists()
        plots=plot_all(root)
        assert len(plots)==14
        assert all(p.exists() for p in plots)
    print('Analysis and plotting smoke tests passed')


if __name__=='__main__':
    main()
