import os
import pandas as pd
from sklearn.metrics import average_precision_score

domain = 'twitter'
ind_data_dir = 'independent/data/' + domain + '/'
fold_dir = 'independent/data/' + domain + '/folds/'
ind_preds_dir = 'independent/output/' + domain + '/predictions/'
rel_preds_dir = 'relational/output/' + domain + '/predictions/'

os.system('cat %stest_*.csv > test.csv' % (fold_dir))
os.system('cat %stest_*_preds.csv > ind_preds.csv' % (ind_preds_dir))
print('cat %smrf_preds_*.csv > mrf_preds.csv' % (rel_preds_dir))
os.system('cat %smrf_preds_*.csv > mrf_preds.csv' % (rel_preds_dir))
print('cat %spsl_preds_*.csv > psl_preds.csv' % (rel_preds_dir))
os.system('cat %spsl_preds_*.csv > psl_preds.csv' % (rel_preds_dir))

labels_df = pd.read_csv(ind_data_dir + 'comments.csv')
ind_df = pd.read_csv('ind_preds.csv')
mrf_df = pd.read_csv('mrf_preds.csv')
psl_df = pd.read_csv('psl_preds.csv')

df = labels_df.merge(ind_df).merge(psl_df).merge(mrf_df)
df = df[['com_id', 'label', 'ind_pred', 'rel_pred', 'mrf_pred']]

# TODO: add tiny bit of noise?

for model in ['ind_pred', 'rel_pred', 'mrf_pred']:
    aupr = average_precision_score(df['label'], df[model])
    print('%s aupr: %.4f' % (model, aupr))
