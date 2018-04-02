"""
Module to test differing featuresets.
"""
import os
import itertools
import pandas as pd


class Relations_Experiment:

    # public
    def __init__(self, config_obj, app_obj):
        self.config_obj = config_obj
        self.app_obj = app_obj

    def run_experiment(self, start=0, end=1000000, domain='twitter',
                       relationsets=['posts', 'intext', 'inhash', 'inment'],
                       clfs=['lr', 'rf', 'xgb'], metric='aupr', engine='psl',
                       fold=0):
        rel_dir = self.config_obj.rel_dir
        out_dir = rel_dir + 'output/' + domain + '/ablation_exp/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        combos = self._create_combinations(relationsets)
        print(combos)

        rows = []
        cols = ['relationset']
        cols.extend(clfs)

        for relationset in combos:
            row = ['+'.join(relationset)]

            for clf in clfs:
                d = self.app_obj.run(domain=domain, start=start, end=end,
                                     fold=fold, engine=engine, clf=clf,
                                     ngrams=True, stacking=0, data='both',
                                     train_size=0.8, val_size=0.01,
                                     relations=relationset)

                row.append(d['psl'][metric])
            rows.append(row)

        fname = engine + '_' + metric + '_rel.csv'
        self._write_scores_to_csv(rows, cols=cols, out_dir=out_dir,
                                  fname=fname)

    # private
    def _clear_data(self, domain='twitter'):
        ind_dir = self.config_obj.ind_dir
        rel_dir = self.config_obj.rel_dir

        fold_dir = ind_dir + '/data/' + domain + '/folds/'
        ind_pred_dir = ind_dir + '/output/' + domain + '/predictions/'
        rel_pred_dir = rel_dir + '/output/' + domain + '/predictions/'

        os.system('rm %s*.csv' % (fold_dir))
        os.system('rm %s*.csv' % (ind_pred_dir))
        os.system('rm %s*.csv' % (rel_pred_dir))

    def _create_combinations(self, fsets):
        all_sets = []

        for L in range(1, len(fsets) + 1):
            for combo in itertools.combinations(fsets, L):
                all_sets.append(list(combo))
        return all_sets

    def _write_scores_to_csv(self, rows, cols=[], out_dir='',
                             fname='results.csv'):
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(out_dir + fname, index=None)
