import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def compute_big_aupr(start_fold=0, num_folds=5, domain='twitter'):
    ind_data_dir = 'independent/data/' + domain + '/'
    ind_preds_dir = 'independent/output/' + domain + '/predictions/'
    rel_preds_dir = 'relational/output/' + domain + '/predictions/'

    ind, mrf, psl = [], [], []

    print('%s: reading model preds for %d folds:' % (domain, num_folds))
    for i in range(start_fold, start_fold + num_folds):
        print('reading preds for fold %d...' % i)

        fname = ind_preds_dir + 'test_' + str(i) + '_preds.csv'
        if os.path.exists(fname):
            ind.append(pd.read_csv(fname))

        fname = rel_preds_dir + 'mrf_preds_' + str(i) + '.csv'
        if os.path.exists(fname):
            mrf.append(pd.read_csv(fname))

        fname = rel_preds_dir + 'psl_preds_' + str(i) + '.csv'
        if os.path.exists(fname):
            psl.append(pd.read_csv(fname))

    print('reading true labels...')
    labels_df = pd.read_csv(ind_data_dir + 'comments.csv')
    labels_df = labels_df[['com_id', 'label']]

    # compute combined test set aupr
    models = []
    df = labels_df
    if len(ind) > 0:
        ind_df = pd.concat(ind)
        df = df.merge(ind_df)
        models.append('ind_pred')
        _compute_mean(ind, df, 'ind_pred')
    if len(mrf) > 0:
        mrf_df = pd.concat(mrf)
        df = df.merge(mrf_df)
        models.append('mrf_pred')
        _compute_mean(mrf, df, 'mrf_pred')
    if len(psl) > 0:
        psl_df = pd.concat(psl)
        df = df.merge(psl_df)
        models.append('psl_pred')
        _compute_mean(psl, df, 'psl_pred')

    print(df.head(5))

    for model in models:
        aupr = average_precision_score(df['label'], df[model])
        print('%s aupr: %.4f' % (model, aupr))


def _compute_mean(model_dfs, df, col='ind_pred'):
    auprs = 0

    for model_df in model_dfs:
        qf = df.merge(model_df)
        aupr = average_precision_score(qf['label'], qf[col])
        if not pd.isnull(aupr) or aupr != 1.0:
            auprs.append(aupr)
    t = (col, len(aupr), np.mean(auprs), np.std(auprs))
    print('%s: num test sets: %d, mean aupr: %.4f +/- %.4f' % t)


def compute_mean_aupr(domain='twitter', data='both'):
    in_dir = 'relational/output/' + domain + '/subsets_exp/'
    path = in_dir + data + '_res.csv'

    df = pd.read_csv(path)
    cols = [x for x in list(df) if '_aupr' in x]
    df = df[cols]

    for col in list(df):
        qf = df[(df[col] != 1.0) & (~pd.isnull(df[col]))]
        t = (col, len(qf), np.mean(qf[col]), np.std(qf[col]))
        print('%s: num test sets: %d, mean aupr: %.4f +/- %.4f' % t)


if __name__ == '__main__':
    domain = 'twitter'
    compute_big_aupr(start_fold=0, num_folds=185, domain=domain)
    compute_mean_aupr(domain=domain, data='both')
