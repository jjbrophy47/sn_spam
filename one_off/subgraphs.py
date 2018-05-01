import argparse
import util as ut
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from generator import Generator


def collect_subgraphs(in_dir='', out_dir='', gid='user_gid', nrows=None):
    gen = Generator()

    ut.out('reading data...', 0)
    df = pd.read_csv(in_dir + 'comments.csv', nrows=nrows)

    ut.out('generating %s...' % gid)
    df = gen.gen_group_id(df, gid)

    ut.out('grouping by %s...' % gid)
    g1 = df.groupby(gid)

    ut.out('computing stats per group...')
    size = g1.size().reset_index().rename(columns={0: 'size'})
    sum_label = g1['label'].sum().reset_index()\
        .rename(columns={'label': 'sum_label'})
    mean_label = g1['label'].mean().reset_index()\
        .rename(columns={'label': 'mean_label'})
    gf = size.merge(sum_label).merge(mean_label)

    ut.out()
    ut.out(str(gf.head(5)))

    same_label = lambda x: 1 if x['mean_label'] in [1.0, 0.0] else 0
    gf['same_label'] = gf.apply(same_label, axis=1)

    ut.out('grouping by size...')
    g2 = gf.groupby('size')

    ut.out('computing stats per size...')
    cnt = g2.size().reset_index().rename(columns={0: 'cnt'})
    slc = g2['same_label'].sum().reset_index()\
        .rename(columns={'same_label': 'same_label_cnt'})
    mean_label2 = g2['mean_label'].mean().reset_index()
    sf = cnt.merge(slc).merge(mean_label2)

    sf['same_label_rto'] = sf['same_label_cnt'] / sf['cnt']

    # compute single node row
    v = gf[gf[gid] == -1][['size', 'mean_label']].values[0]
    row = [(1, v[0], v[0], v[1], 1.0)]
    cols = list(sf)
    sfs = pd.DataFrame(row, columns=cols)
    sf = pd.concat([sfs, sf])

    ut.out()
    ut.out(str(sf.head(5)))

    # keep top X% of affected nodes
    pct = 99
    total = sf.cnt.sum()
    for i in range(1, len(gf)):
        if sf[:i].cnt.sum() / total >= pct / float(100):
            sf = sf[:i]
            break

    ut.out('plotting...')
    cols = ['cnt', 'same_label_rto', 'mean_label']
    ncols = len(cols)

    nrows = 2
    ncols = int(ncols / nrows)
    ncols += 1 if ncols / nrows != 0 else 0

    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15))
    axs = axs.flatten()
    for i, col in enumerate(cols):
        sf.plot.barh('size', col, ax=axs[i], title=col, legend=False,
                     fontsize=8)

    fig.tight_layout()
    fig.suptitle('subgraph stats, top %d%% of nodes' % pct, y=1.01)
    fig.savefig(out_dir + 'sg_%s.pdf' % gid, format='pdf',
                bbox_inches='tight')
    plt.close('all')

    # sf.plot.barh('size', 'cnt')
    # sf.plot.barh('size', 'same_label_rto')
    # sf.plot.barh('size', 'mean_label')
    # plt.show()
    ut.out()


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
    parser.add_argument('-n', metavar='NUM', type=int, default=100000000,
                        help='nrows, default: %(default)s')
    parser.add_argument('--gids', nargs='*', metavar='GID',
                        help='list of gids, default: %(default)s')
    args = parser.parse_args()

    in_dir = 'independent/data/' + args.d + '/'
    out_dir = 'relational/output/' + args.d + '/subgraphs/'
    fold = args.f
    nrows = args.n
    gids = args.gids if args.gids is not None else []

    ut.makedirs(out_dir)

    # analyze_subgraphs(in_dir=in_dir, out_dir=in_dir, fold=fold)

    for gid in gids:
        collect_subgraphs(in_dir, out_dir, gid=gid, nrows=nrows)
