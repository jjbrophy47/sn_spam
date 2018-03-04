import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

exp_start = 0
num_exps = 200
domain = 'youtube'

ind_data_dir = 'independent/data/' + domain + '/'
fold_dir = 'independent/data/' + domain + '/folds/'
ind_preds_dir = 'independent/output/' + domain + '/predictions/'
rel_preds_dir = 'relational/output/' + domain + '/predictions/'

ind = []
mrf = []
psl = []
for i in range(exp_start, exp_start + num_exps):
    fname = ind_preds_dir + 'test_' + str(i) + '_preds.csv'
    if os.path.exists(fname):
        ind.append(pd.read_csv(fname))

    fname = rel_preds_dir + 'mrf_preds_' + str(i) + '.csv'
    if os.path.exists(fname):
        mrf.append(pd.read_csv(fname))

    fname = rel_preds_dir + 'psl_preds_' + str(i) + '.csv'
    if os.path.exists(fname):
        psl.append(pd.read_csv(fname))

labels_df = pd.read_csv(ind_data_dir + 'comments.csv')
labels_df = labels_df[['com_id', 'label']]

ind_df = pd.concat(ind)
mrf_df = pd.concat(mrf)
psl_df = pd.concat(psl)

print(df)

# TODO: add tiny bit of noise?

def compute_mean_aupr(dfs, labels_df, model='ind_pred'):
    auprs = []
    for sample_df in dfs:
        df = labels_df.merge(sample_df)
        aupr = average_precision_score(df['label'], df[model])
        auprs.append(aupr)
    t = (model, len(dfs), np.mean(auprs), np.std(auprs))
    print('%s: num test sets: %d, mean aupr: %.4f +/- %.4f' % t)

l = [(ind_df, 'ind_pred', mrf_df, 'mrf_pred', psl_df, 'psl_pred')]
for dfs, model in l:
    compute_mean_aupr(dfs, labels_df, model=model)

# compute combined test set aupr
df = labels_df.merge(ind_df).merge(mrf_df).merge(psl_df)
for model in ['ind_pred', 'rel_pred', 'mrf_pred']:
    aupr = average_precision_score(df['label'], df[model])
    print('%s aupr: %.4f' % (model, aupr))
