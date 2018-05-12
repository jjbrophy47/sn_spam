import argparse
import util as ut
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from generator import Generator
from connections import Connections


def multi_relational(in_dir='', out_dir='', gids=['text_gid'], pts=100000,
                     dom=''):
    con = Connections()
    gen = Generator()

    ut.out('gids: %s' % str(gids), 0)

    t1 = ut.out('reading data...')
    df = pd.read_csv(in_dir + 'comments.csv', nrows=pts)
    pts = len(df)
    ut.time(t1)

    rels = []
    for gid in gids:
        df = gen.gen_group_id(df, gid)
        g = gid.replace('_gid', '')
        rel = 'has' + g
        rels.append((rel, g, gid))

    g, sgs = con.find_subgraphs(df, rels)

    t1 = ut.out('computing stats per group...')
    cols = ['size', 'mean_label', 'same_label']
    rows = []
    for msg_nodes, hub_nodes, rels, edges in sgs:
        size = len(msg_nodes)
        labels = [int(x.split('+')[1]) for x in msg_nodes]
        mean_label = np.mean(labels)
        same_label = 1 if mean_label in [1.0, 0.0] else 0
        rows.append((size, mean_label, same_label))
    gf = pd.DataFrame(rows, columns=cols)
    ut.time(t1)

    t1 = ut.out('grouping by size...')
    g2 = gf.groupby('size')
    ut.time(t1)

    t1 = ut.out('computing stats per size...')
    cnt = g2.size().reset_index().rename(columns={0: 'cnt'})
    slc = g2['same_label'].sum().reset_index()\
        .rename(columns={'same_label': 'same_label_cnt'})
    mean_label2 = g2['mean_label'].mean().reset_index()
    sf = cnt.merge(slc).merge(mean_label2)

    sf['cnt_rto'] = (sf['cnt'] * sf['size']) / len(df)
    sf['same_label_rto'] = sf['same_label_cnt'] / sf['cnt']

    # keep top X% of affected nodes
    pct = 100
    total = sf.cnt.sum()
    for i in range(1, len(sf)):
        if sf[:i].cnt.sum() / total >= pct / float(100):
            sf = sf[:i]
            break
    ut.time(t1)

    t1 = ut.out('plotting...')
    cols = ['cnt_rto', 'same_label_rto', 'mean_label']
    ncols = len(cols)

    nrows = 2
    ncols = int(ncols / nrows)
    ncols += 1 if ncols / nrows != 0 else 0

    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15))
    axs = axs.flatten()
    for i, col in enumerate(cols):
        sf.plot.barh('size', col, ax=axs[i], title=col, legend=False,
                     fontsize=8)

    title = '%s: %d data points, top %d%%' % (dom, pts, pct)
    fig.tight_layout()
    fig.suptitle(title, y=1.01)
    fig.savefig(out_dir + 'sg_%s.pdf' % str(gids), format='pdf',
                bbox_inches='tight')
    plt.close('all')
    ut.time(t1)
    ut.out()


def single_relational(in_dir='', out_dir='', gids=['text_gid'], pts=100000,
                      start=0, dom=''):
    gen = Generator()

    ut.out('gids: %s' % str(gids), 0)

    t1 = ut.out('reading data...')
    df = pd.read_csv(in_dir + 'comments.csv', skiprows=range(1, start),
                     nrows=pts)
    pts = len(df)
    ut.time(t1)

    for gid in gids:
        t1 = ut.out('generating %s...' % gid)
        df = gen.gen_group_id(df, gid)
        ut.time(t1)

    for gid in gids:
        t1 = ut.out('grouping by %s...' % gid)
        g1 = df.groupby(gid)
        ut.time(t1)

        t1 = ut.out('computing stats per group...')
        size = g1.size().reset_index().rename(columns={0: 'size'})
        sum_label = g1['label'].sum().reset_index()\
            .rename(columns={'label': 'sum_label'})
        mean_label = g1['label'].mean().reset_index()\
            .rename(columns={'label': 'mean_label'})
        gf = size.merge(sum_label).merge(mean_label)

        single_cnt = gf[gf[gid] == -1]['size'].values[0]

        same_label = lambda x: 1 if x['mean_label'] in [1.0, 0.0] else 0
        gf['same_label'] = gf.apply(same_label, axis=1)
        ut.time(t1)

        t1 = ut.out('grouping by size...')
        g2 = gf.groupby('size')
        ut.time(t1)

        t1 = ut.out('computing stats per size...')
        cnt = g2.size().reset_index().rename(columns={0: 'cnt'})
        slc = g2['same_label'].sum().reset_index()\
            .rename(columns={'same_label': 'same_label_cnt'})
        mean_label2 = g2['mean_label'].mean().reset_index()
        sf = cnt.merge(slc).merge(mean_label2)

        sf['cnt_rto'] = (sf['cnt'] * sf['size']) / len(df)
        sf['same_label_rto'] = sf['same_label_cnt'] / sf['cnt']

        sf = sf[sf['size'] != single_cnt]

        # compute mean label for groups that do/do not have the same labels.
        gfs = gf[gf['same_label'] == 1]
        gfo = gf[gf['same_label'] == 0]
        g2s = gfs.groupby('size')
        g2o = gfo.groupby('size')
        sfs_df = g2s['mean_label'].mean().reset_index()\
            .rename(columns={'mean_label': 'mean_label_same_label'})
        sfo_df = g2o['mean_label'].mean().reset_index()\
            .rename(columns={'mean_label': 'mean_label_not_same_label'})

        # compute single node row
        sfo_df = sfo_df[sfo_df['size'] != single_cnt]
        vo = gfo[gfo[gid] == -1][['size', 'mean_label']].values[0]
        row = [(1, vo[1])]
        sfs = pd.DataFrame(row, columns=list(sfo_df))
        sfo_df = pd.concat([sfs, sfo_df])

        # compute single node row
        v = gf[gf[gid] == -1][['size', 'mean_label']].values[0]
        row = [(1, v[0], v[0], v[1], v[0] / len(df), 1.0)]
        cols = list(sf)
        sfs = pd.DataFrame(row, columns=cols)
        sf = pd.concat([sfs, sf])

        # keep top X% of affected nodes
        pct = 100
        total = sf.cnt.sum()
        for i in range(1, len(sf)):
            if sf[:i].cnt.sum() / total >= pct / float(100):
                sf = sf[:i]
                break
        ut.time(t1)

        t1 = ut.out('plotting...')
        cols = ['cnt_rto', 'same_label_rto',
                'mean_label_same_label', 'mean_label_not_same_label']
        ncols = len(cols)

        nrows = 2
        ncols = int(ncols / nrows)
        ncols += 1 if ncols % nrows != 0 else 0

        fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15))
        axs = axs.flatten()
        for i, col in enumerate(cols):
            if col == 'mean_label_same_label':
                dummy_df = sfs_df
            elif col == 'mean_label_not_same_label':
                dummy_df = sfo_df
            else:
                dummy_df = sf

            if len(dummy_df) > 0:
                dummy_df.plot.barh('size', col, ax=axs[i], title=col,
                                   legend=False, fontsize=8)

        title = '%s: %d data points, top %d%%' % (dom, pts, pct)
        fig.tight_layout()
        fig.suptitle(title, y=1.01)
        fig.savefig(out_dir + 'sg_%s.pdf' % str(gid), format='pdf',
                    bbox_inches='tight')
        plt.close('all')
        ut.time(t1)

    spam_rto = df.label.sum() / len(df)
    ut.out('spam ratio: %.2f' % spam_rto)

    if len(gids) > 1:
        rel_nodes = 0
        g = df.groupby(gids).size().reset_index().rename(columns={0: 'size'})
        for gid in gids:
            g = g[g[gid] != -1]
            rel_nodes += len(df[df[gid] != -1])

        spam_rto = df.label.sum() / len(df)
        overlap_rto = g.size.sum() / rel_nodes
        ut.out('overlap ratio: %.2f' % overlap_rto)
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
    parser.add_argument('-s', metavar='NUM', type=int, default=0,
                        help='start, default: %(default)s')
    parser.add_argument('-n', metavar='NUM', type=int, default=100000000,
                        help='nrows, default: %(default)s')
    parser.add_argument('--gids', nargs='*', metavar='GID',
                        help='list of gids, default: %(default)s')
    args = parser.parse_args()

    domain = args.d
    start = args.s
    nrows = args.n
    gids = args.gids if args.gids is not None else []

    in_dir = 'independent/data/' + domain + '/'
    out_dir = 'relational/output/' + domain + '/subgraphs/'
    ut.makedirs(out_dir)

    # analyze_subgraphs(in_dir=in_dir, out_dir=in_dir, fold=fold)

    single_relational(in_dir, out_dir, gids=gids, pts=nrows, start=start,
                      dom=domain)
    # multi_relational(in_dir, out_dir, gids=gids, pts=nrows, dom=domain)
