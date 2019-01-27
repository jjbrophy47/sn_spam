"""
This script uses EGGS to model a wikipedia spam dataset.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from eggs.model import EGGS
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from features.wikipedia import pseudo_relational as pr

from collections import defaultdict
from eggs import print_utils

data_dir = 'data/wikipedia/processed/'
data_df = pd.read_csv('%sdata.csv' % data_dir)

X_cols = list(data_df.columns)
X_cols.remove('user_id')
X_cols.remove('label')

data_df = data_df.fillna(0)
data_df = data_df.sample(frac=1, random_state=77)

X = data_df[X_cols].to_numpy()
y = data_df['label'].to_numpy()
target_col = data_df['user_id'].to_numpy()

lr = LogisticRegression(solver='liblinear')
rf = RandomForestClassifier()
xgb = XGBClassifier()
lgb = LGBMClassifier()

metrics = [('roc_auc', roc_auc_score), ('aupr', average_precision_score), ('accuracy', accuracy_score)]
models = ['eggs']

scores = defaultdict(list)
all_scores = defaultdict(list)
predictions = y.copy().astype(float)
predictions_binary = y.copy().astype(float)

kf = KFold(n_splits=10, random_state=69, shuffle=True)
for fold, (train_index, test_index) in enumerate(kf.split(X)):

    print('\nfold %d...' % fold)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    target_col_train, target_col_test = target_col[train_index], target_col[test_index]

    print('fitting...')
    eggs = EGGS(estimator=xgb)
    # eggs = EGGS(estimator=xgb, sgl_method='cv', stacks=2, joint_model='psl', relations=['link_id'],
    #             pr_func=pr.pseudo_relational_features, verbose=1)
    eggs.fit(X_train, y_train, target_col_train)

    print('predicting...')
    y_hat = eggs.predict_proba(X_test, target_col_test)[:, 1]
    y_hat_binary = eggs.predict(X_test, target_col_test)

    np.put(predictions, test_index, y_hat)
    np.put(predictions, test_index, y_hat_binary)

    for metric, scorer in metrics:
        score = scorer(y_test, y_hat_binary) if metric == 'accuracy' else scorer(y_test, y_hat)
        scores['eggs' + '|' + metric].append(score)

# compute single score using predictions from all folds
for metric, scorer in metrics:
    score = scorer(y, predictions_binary) if metric == 'accuracy' else scorer(y, predictions)
    all_scores['eggs' + '|' + metric].append(score)

print_utils.print_scores(scores, models, metrics)
print_utils.print_scores(all_scores, models, metrics)
# eggs.plot_feature_importance(X_cols)
