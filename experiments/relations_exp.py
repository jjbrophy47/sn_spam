"""
Module to test differing featuresets.
"""
import os
import itertools
import pandas as pd


class Relations_Experiment:

    # public
    def __init__(self, config_obj, app_obj, util_obj):
        self.config_obj = config_obj
        self.app_obj = app_obj
        self.util_obj = util_obj

    def run_experiment(self, start=0, end=1000000, domain='twitter',
                       relationsets=['posts', 'intext', 'inhash', 'inment'],
                       clf='lgb', metric='aupr', engine='psl',
                       fold=0, train_size=0.8, val_size=0.1):

        rel_dir = self.config_obj.rel_dir
        out_dir = rel_dir + 'output/' + domain + '/experiments/'
        self.util_obj.create_dirs(out_dir)

        combos = self._create_combinations(relationsets)
        print(combos)

        rows = []
        cols = ['relationset', clf]
        if engine in ['psl', 'all']:
            cols.append('psl')
        if engine in ['mrf', 'all']:
            cols.append('mrf')

        for relationset in combos:
            row = ['+'.join(relationset)]

            d = self.app_obj.run(domain=domain, start=start, end=end,
                                 fold=fold, engine=engine, clf=clf,
                                 ngrams=True, stacking=0, data='both',
                                 train_size=train_size, val_size=val_size,
                                 relations=relationset)

            for model in ['ind', 'psl', 'mrf']:
                if d.get(model) is not None:
                    row.append(d[model][metric])
            rows.append(row)

        fn = engine + '_' + metric + '_rel.csv'
        self._write_scores_to_csv(rows, cols=cols, out_dir=out_dir, fname=fn)

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
