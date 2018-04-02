"""
Module to test increasing levels of stacking.
"""
import os
import pandas as pd


class Stacking_Experiment:

    # public
    def __init__(self, config_obj, app_obj, util_obj):
        self.config_obj = config_obj
        self.app_obj = app_obj
        self.util_obj = util_obj

    def run_experiment(self, start=0, end=2000000, domain='twitter',
                       clfs=['lr', 'rf', 'xgb'], start_stack=0, end_stack=7,
                       relations=[], metric='aupr', fold=0, train_size=0.8):
        """Configures the application based on the data subsets, and then runs
                the independent and relational models."""
        assert end_stack >= start_stack

        rel_dir = self.config_obj.rel_dir
        out_dir = rel_dir + 'output/' + domain + '/experiments/'
        self.util_obj.create_dirs(out_dir)

        rows = []
        cols = ['stacks']
        cols.extend(clfs)

        for i, stacks in enumerate(range(start_stack, end_stack + 1)):
            row = [stacks]

            for clf in clfs:
                d = self.app_obj.run(domain=domain, start=start, end=end,
                                     fold=fold, engine=None, clf=clf,
                                     ngrams=False, stacking=stacks,
                                     data='both', train_size=train_size,
                                     val_size=0, relations=relations)

                row.append(d['ind'][metric])
            rows.append(row)

        fn = metric + '_ind_stacks.csv'
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

    def _write_scores_to_csv(self, rows, cols=[], out_dir='',
                             fname='results.csv'):
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(out_dir + fname, index=None)
