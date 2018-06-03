import os
import argparse
import util as ut
import pandas as pd
from generator import Generator
from connections import Connections
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve


def compute_big_aupr(start_fold=0, ref_start_fold=-1, num_folds=5,
                     domain='twitter', models=['ind'], in_dir='', gids=[]):
    ind_data_dir = 'independent/data/' + domain + '/'

    lines = {'ind': 'b-', 'mrf': 'g--', 'psl': 'm-.', 'mean': 'r:',
             'median': 'c:', 'max': 'y:'}
    inds, mrfs, psls, approxs, refs = [], [], [], [], []
    preds = []

    gen_obj = Generator()
    relations = _relations_for_gids(gids)

    for model in models:
        preds.append(model + '_pred')
    if 'approx' in models:
        models.remove('approx')
        models.extend(['mean', 'median', 'max'])
        preds.extend(['mean_pred', 'median_pred', 'max_pred'])
    preds = list(zip(models, preds))

    ut.out('reading true labels...')
    full_df = pd.read_csv(ind_data_dir + 'comments.csv')

    s = '%s: reading model preds from fold %d to %d:'
    ut.out(s % (domain, start_fold, start_fold + num_folds - 1), 0)

    for i in range(start_fold, start_fold + num_folds):
        ut.out('reading preds for fold %d...' % i)

        if ref_start_fold > -1:
            ndx = ref_start_fold + i
            fname = in_dir + 'test_' + str(i) + '_preds.csv'
            assert os.path.exists(fname)
            refs.append(pd.read_csv(fname))

        if 'ind' in models:
            fname = in_dir + 'test_' + str(i) + '_preds.csv'
            assert os.path.exists(fname)
            ind_df = pd.read_csv(fname)
            inds.append(ind_df)

            if 'mean' in models:
                temp_df = full_df[full_df['com_id'].isin(ind_df['com_id'])]

                for gid in gids:
                    t1 = ut.out('generating %s...' % gid)
                    temp_df = gen_obj.gen_group_id(temp_df, gid)
                    ut.time(t1)

                approx_df = _approximations(temp_df, relations)
                approxs.append(approx_df)

        if 'mrf' in models:
            fname = in_dir + 'mrf_preds_' + str(i) + '.csv'
            assert os.path.exists(fname)
            mrfs.append(pd.read_csv(fname))

        if 'psl' in models:
            fname = in_dir + 'psl_preds_' + str(i) + '.csv'
            assert os.path.exists(fname)
            psls.append(pd.read_csv(fname))

    df = full_df[['com_id', 'label']]

    if 'ind' in models:
        ind_df = pd.concat(inds)
        df = df.merge(ind_df)

        if 'meean' in models:
            approx_df = pd.concat(approxs)
            assert set(ind_df['com_id']) == set(approx_df['com_id'])
            df = df.merge(approx_df)

    if ref_start_fold > -1:
        ref_df = pd.concat(refs)
        ref_df = full_df[['com_id', 'label']].merge(ref_df)
        assert set(ind_df['com_id']) == set(ref_df['com_id'])

    if 'mrf' in models:
        mrf_df = pd.concat(mrfs)
        assert set(ind_df['com_id']) == set(mrf_df['com_id'])
        df = df.merge(mrf_df)

    if 'psl' in models:
        psl_df = pd.concat(psls)
        assert set(ind_df['com_id']) == set(psl_df['com_id'])
        df = df.merge(psl_df)


    # compute reference aupr and auroc
    ref_label, ref_pred = ref_df['label'], ref_df['ind_pred']
    ref_aupr = average_precision_score(ref_label, ref_pred)
    ref_auroc = roc_auc_score(ref_label, ref_pred)
    ref_p, ref_r, ref_t = precision_recall_curve(ref_label, ref_pred)
    ref_fpr, ref_tpr, ref_t2 = roc_curve(ref_label, ref_pred)
    ut.out('%s aupr: %.4f, auroc: %.4f' % ('reference', ref_aupr, ref_auroc))

    # compute combined test set curves
    for i, (model, pred) in enumerate(preds):
        aupr = average_precision_score(df['label'], df[pred])
        auroc = roc_auc_score(df['label'], df[pred])
        p, r, _ = precision_recall_curve(df['label'], df[pred])
        fpr, tpr, _ = roc_curve(df['label'], df[pred])
        _, aupr_pval = _significance(ref_r, ref_p, r, p)
        _, auroc_pval = _significance(ref_fpr, ref_tpr, fpr, tpr)
        t = (model, aupr, aupr_pval, auroc, auroc_pval)
        ut.out('%s aupr: %.4f (%.4f), auroc: %.4f (%.4f)' % t)

        save = True if i == len(preds) - 1 else False
        ut.plot_pr_curve(model, p, r, aupr, save=save, domain=domain,
                         line=lines[model], show_legend=True)
        ut.plot_roc_curve(model, tpr, fpr, auroc, save=save, domain=domain,
                          line=lines[model], show_legend=True)
    ut.out()

def _approximations(df, relations):
    t1 = ut.out('approximating relational with mean, max, median...')

    con_obj = connections.Connections()
    g, sgs = con_obj.find_subgraphs(df, relations)
    approx_dict = {}

    sg_list = []
    for i, sg in enumerate(sgs):
        if sg[3] > 0:  # num edges > 0
            sg_list.extend([(x, i) for x in sg[0]])  # give sg_id

    if len(sg_list) == 0:
        return approx_dict

    sg_df = pd.DataFrame(sg_list, columns=['com_id', 'sg_id'])
    df = df.merge(sg_df, how='left')
    df['sg_id'] = df['sg_id'].fillna(-1).apply(int)

    sg_mean = df.groupby('sg_id')['ind_pred'].mean().reset_index()\
        .rename(columns={'ind_pred': 'mean_pred'})
    sg_median = df.groupby('sg_id')['ind_pred'].median().reset_index()\
        .rename(columns={'ind_pred': 'median_pred'})
    sg_max = df.groupby('sg_id')['ind_pred'].max().reset_index()\
        .rename(columns={'ind_pred': 'max_pred'})
    df = df.merge(sg_mean).merge(sg_median).merge(sg_max)

    ut.time(t1)
    return df[['com_id', 'mean_pred', 'median_pred', 'max_pred']]

def _relations_for_gids(gids):
    relations = []
    for gid in gids:
        group = g_id.replace('_gid', '')
        rel = 'has' + group
        relations.append((rel, group, gid))
    return relations

def _significance(ref_x, ref_y, target_x, target_y, num_samples=20):
    f_ref = interp1d(ref_x, ref_y)
    f_target = interp1d(target_x, target_y)

    samples = np.linspace(0.0, 1.0, num_samples)

    ref_yvals = [f_ref(x).item(0) for x in samples]
    target_yvals = [f_target(x).item(0) for x in samples]

    t_stat, pval = ttest_rel(ref_yvals, target_yvals)
    return pval


if __name__ == '__main__':
    description = 'Script to merge and compute subset predictions scores'
    parser = argparse.ArgumentParser(description=description, prog='big_aupr')

    parser.add_argument('-d', metavar='DOMAIN',
                        help='domain, default: %(default)s')
    parser.add_argument('--start_fold', metavar='NUM', type=int,
                        help='first subset, default: %(default)s')
    parser.add_argument('--ref_start_fold', metavar='NUM', type=int, default=-1,
                        help='references first subset, default: %(default)s')
    parser.add_argument('--num_folds', metavar='NUM', type=int,
                        help='number of subsets, default: %(default)s')
    parser.add_argument('--models', nargs='*', metavar='MODEL',
                        help='list of models, default: %(default)s')
    parser.add_argument('--in_dir', metavar='DIR', default='',
                        help='predictions directory, default: %(default)s')
    parser.add_argument('--gids', nargs='*', metavar='GID',
                        help='list of gids, default: %(default)s')

    args = parser.parse_args()
    domain = args.d
    start = args.start_fold
    ref_start = args.ref_start_fold
    folds = args.num_folds
    models = args.models if args.models is not None else ['ind']
    in_dir = args.in_dir
    gids = args.gids if args.gids is not None else []

    compute_big_aupr(start_fold=start, ref_start_fold=ref_start,
                     num_folds=folds, domain=domain, models=models,
                     in_dir=in_dir, gids=gids)
