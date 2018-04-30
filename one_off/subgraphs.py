import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import reduce


def analyze_subgraphs(in_dir='', out_dir='', fold=0):
    print('analyzing connected components...')

    df = pd.read_csv(in_dir + 'sg_stats_%s.csv' % fold)
    g = df.groupby('size')

    gcnt = g.size().reset_index().rename(columns={0: 'cnt'})
    gsame = g['same'].sum().reset_index()
    gmx = g['max'].mean().reset_index()
    gmn = g['min'].mean().reset_index()
    gsd = g['std'].mean().reset_index()
    gmean = g['mean'].mean().reset_index()
    gmedian = g['median'].mean().reset_index()
    glab_mean = g['lab_mean'].mean().reset_index()
    glab_diff = g['lab_diff'].mean().reset_index()

    dfl = [gcnt, gsame, gmx, gmn, gsd, gmean, gmedian,
           glab_mean, glab_diff]
    gf = reduce(lambda x, y: pd.merge(x, y, on='size'), dfl)

    # derived statistics
    gf['same_rto'] = gf['same'].divide(gf['cnt'])
    gf['affected'] = (gf.cnt - gf.same) * gf['size']

    gf = gf.sort_values('affected', ascending=False)

    # keep top X% of affected nodes
    pct = 75
    total_affected = gf.affected.sum()
    for i in range(1, len(gf)):
        if gf[:i].affected.sum() / total_affected >= pct / float(100):
            gf = gf[:i]
            break

    # plot graphs
    rm = ['size', 'same']
    cols = list(gf)
    cols = [x for x in cols if x not in rm]
    ncols = len(cols)

    nrows = 4
    ncols = int(ncols / nrows)
    ncols += 1 if ncols / nrows != 0 else 0

    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 20))
    axs = axs.flatten()
    for i, col in enumerate(cols):
        gf.plot.barh('size', col, ax=axs[i], title=col, legend=False,
                     fontsize=8)

    fig.tight_layout()
    fig.suptitle('subgraph stats, top %d%% of affected nodes' % pct, y=1.01)
    fig.savefig(out_dir + 'sg_plots_%s.pdf' % fold, format='pdf',
                bbox_inches='tight')
    plt.close('all')

if __name__ == '__main__':
    description = 'Script to analyze connected components'
    parser = argparse.ArgumentParser(description=description, prog='subgraphs')

    parser.add_argument('-d', metavar='DOMAIN',
                        help='domain, default: %(default)s')
    parser.add_argument('-f', metavar='NUM', type=int,
                        help='first subset, default: %(default)s')
    args = parser.parse_args()

    in_dir = 'relational/output/' + args.d + '/subgraphs/'
    fold = args.f

    analyze_subgraphs(in_dir=in_dir, out_dir=in_dir, fold=fold)
