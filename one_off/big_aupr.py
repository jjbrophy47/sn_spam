# import os
import pandas as pd
from sklearn.metrics import average_precision_score

num_experiments = 200
domain = 'youtube'

ind_data_dir = 'independent/data/' + domain + '/'
fold_dir = 'independent/data/' + domain + '/folds/'
ind_preds_dir = 'independent/output/' + domain + '/predictions/'
rel_preds_dir = 'relational/output/' + domain + '/predictions/'

# os.system('cat %stest_*.csv > test.csv' % (fold_dir))
# os.system('cat %stest_*_preds.csv > ind_preds.csv' % (ind_preds_dir))
# os.system('cat %smrf_preds_*.csv > mrf_preds.csv' % (rel_preds_dir))
# os.system('cat %spsl_preds_*.csv > psl_preds.csv' % (rel_preds_dir))

ind = []
mrf = []
psl = []
for i in range(num_experiments):
    ind.append(pd.read_csv(ind_preds_dir + 'test_' + str(i) + '_preds.csv'))
    mrf.append(pd.read_csv(rel_preds_dir + 'mrf_preds_' + str(i) + '_.csv'))
    psl.append(pd.read_csv(rel_preds_dir + 'psl_preds_' + str(i) + '_.csv'))

labels_df = pd.read_csv(ind_data_dir + 'comments.csv')
labels_df = labels_df[['com_id', 'label']]
ind_df = pd.concat(ind)
mrf_df = pd.concat(mrf)
psl_df = pd.concat(psl)

df = labels_df.merge(ind_df).merge(mrf_df).merge(psl_df)

# q1 = labels_df.merge(ind_df, on='com_id')
# q2 = q1.merge(mrf_df, on='com_id')
# df = q2.merge(psl_df, on='com_id')
# df = df[['com_id', 'label', 'ind_pred', 'rel_pred', 'mrf_pred']]

print(df)

# TODO: add tiny bit of noise?

for model in ['ind_pred', 'rel_pred', 'mrf_pred']:
    aupr = average_precision_score(df['label'], df[model])
    print('%s aupr: %.4f' % (model, aupr))
