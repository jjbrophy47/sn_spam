"""
Module to test increasing levels of stacking.
"""
import os
import pandas as pd


class Stacking_Experiment:

    # public
    def __init__(self, config_obj, app_obj):
        self.config_obj = config_obj
        self.app_obj = app_obj

    def run_experiment(self, start=0, end=2000000, domain='twitter',
                       clfs=['lr', 'rf', 'xgb'], num_stacks=2,
                       relations=[]):
        """Configures the application based on the data subsets, and then runs
                the independent and relational models."""
        assert num_stacks > 0

        rel_dir = self.config_obj.rel_dir
        out_dir = rel_dir + 'output/' + domain + '/stacking_exp/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        rows = []
        cols = ['stacks']
        cols.extend(clfs)

        for i, stacks in enumerate(range(num_stacks)):
            row = [stacks]

            for clf in clfs:
                d = self.app_obj.run(domain=domain, start=start, end=end,
                                     fold=1000, engine=None, clf=clf,
                                     ngrams=False, stacking=stacks,
                                     data='both', train_size=0.9,
                                     val_size=0, relations=relations)

                row.append(d['ind']['auroc'])
            rows.append(row)

        self._write_scores_to_csv(rows, cols=cols, out_dir=out_dir,
                                  fname='ind_stacks.csv')

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
