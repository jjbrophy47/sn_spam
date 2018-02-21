import os
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

df = labels_df.merge(ind_df).merge(mrf_df).merge(psl_df)

print(df)

# TODO: add tiny bit of noise?

for model in ['ind_pred', 'rel_pred', 'mrf_pred']:
    aupr = average_precision_score(df['label'], df[model])
    print('%s aupr: %.4f' % (model, aupr))
