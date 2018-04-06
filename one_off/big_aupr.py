import os
import argparse
import util as ut
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def compute_big_aupr(start_fold=0, num_folds=5, domain='twitter'):
    ind_data_dir = 'independent/data/' + domain + '/'
    ind_preds_dir = 'independent/output/' + domain + '/predictions/'
    rel_preds_dir = 'relational/output/' + domain + '/predictions/'

    ind, mrf, psl = [], [], []

    s = '%s: reading model preds from fold %d to %d:'
    ut.out(s % (domain, start_fold, num_folds - 1), 0)

    for i in range(start_fold, start_fold + num_folds):
        ut.out('reading preds for fold %d...' % i)

        fname = ind_preds_dir + 'test_' + str(i) + '_preds.csv'
        if os.path.exists(fname):
            ind.append(pd.read_csv(fname))

        fname = rel_preds_dir + 'mrf_preds_' + str(i) + '.csv'
        if os.path.exists(fname):
            mrf.append(pd.read_csv(fname))

        fname = rel_preds_dir + 'psl_preds_' + str(i) + '.csv'
        if os.path.exists(fname):
            psl.append(pd.read_csv(fname))

    ut.out('reading true labels...')
    labels_df = pd.read_csv(ind_data_dir + 'comments.csv')
    labels_df = labels_df[['com_id', 'label']]

    # compute combined test set aupr
    models = []
    df = labels_df

    if len(ind) > 0:
        ind_df = pd.concat(ind)
        df = df.merge(ind_df)
        print(df.head(5), len(df))
        models.append('ind_pred')
        _compute_mean(ind, df, 'ind_pred')

    # if len(mrf) > 0:  # TODO: mrf not predicting all values.
    #     mrf_df = pd.concat(mrf)
    #     df = df.merge(mrf_df)
    #     print(df.head(5), len(df))
    #     models.append('mrf_pred')
    #     _compute_mean(mrf, df, 'mrf_pred')

    if len(psl) > 0:
        psl_df = pd.concat(psl)
        df = df.merge(psl_df)
        print(df.head(5), len(df))
        models.append('psl_pred')
        _compute_mean(psl, df, 'psl_pred')

    for model in models:
        aupr = average_precision_score(df['label'], df[model])
        ut.out('%s aupr: %.4f' % (model, aupr))
    ut.out()


def _compute_mean(model_dfs, df, col='ind_pred'):
    auprs = []

    ut.out('computing mean aupr for %s:' % col)
    for i, model_df in enumerate(model_dfs):
        ut.out('%d...' % i, 0)
        qf = df.merge(model_df)
        # print(qf)
        aupr = average_precision_score(qf['label'], qf[col])
        if not pd.isnull(aupr) or aupr != 1.0:
            auprs.append(aupr)
    t = (col, len(auprs), np.mean(auprs), np.std(auprs))
    ut.out('%s: num test sets: %d, mean aupr: %.4f +/- %.4f' % t)


if __name__ == '__main__':
    description = 'Script to merge and compute subset predictions scores'
    parser = argparse.ArgumentParser(description=description, prog='big_aupr')

    parser.add_argument('-d', metavar='DOMAIN',
                        help='domain, default: %(default)s')
    parser.add_argument('--start_fold', metavar='NUM', type=int,
                        help='first subset, default: %(default)s')
    parser.add_argument('--num_folds', metavar='NUM', type=int,
                        help='number of subsets, default: %(default)s')

    args = parser.parse_args()
    compute_big_aupr(start_fold=args.start_fold, num_folds=args.num_folds,
                     domain=args.d)
