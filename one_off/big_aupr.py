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
    ind_df = pd.concat(ind)
    mrf_df = pd.concat(mrf)
    psl_df = pd.concat(psl)
    df = labels_df.merge(ind_df).merge(mrf_df).merge(psl_df)

    for model in ['ind_pred', 'psl_pred', 'mrf_pred']:
        aupr = average_precision_score(df['label'], df[model])
        print('%s aupr: %.4f' % (model, aupr))


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
    compute_big_aupr(start_fold=0, num_folds=5, domain=domain)
    compute_mean_aupr(domain=domain, data='both')
