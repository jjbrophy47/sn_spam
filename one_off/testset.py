import os
import argparse
import util as ut
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# r('\\u\S\S\S\S')  unicode extractor

df = pd.read_csv('')
ip = pd.read_csv('')  # ind_preds
pp = pd.read_csv('')  # psl_preds

qf = df.merge(ip).merge(pp)

qf['ind_pred'] = qf['ind_pred'] / qf['ind_pred'].max()
qf['psl_pred'] = qf['psl_pred'] / qf['psl_pred'].max()

qf['ind_rank'] = qf['ind_pred'].rank(method='first').apply(int)
qf['psl_rank'] = qf['psl_pred'].rank(method='first').apply(int)
qf['inv_lbl'] = qf['label'].apply(lambda x: 1 if x == 0 else 0)

qf = qf.sort_values('ind_rank', ascending=False)
qf['ind_ham_sum'] = qf['inv_lbl'].cumsum()

qf = qf.sort_values('psl_rank', ascending=False)
qf['psl_ham_sum'] = qf['inv_lbl'].cumsum()

qf['ham_sum_diff'] = qf['ind_ham_sum'] - qf['psl_ham_sum']


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
