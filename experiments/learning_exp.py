"""
Module to generate learning curves.
"""
import os
import pandas as pd


class Learning_Experiment:

    # public
    def __init__(self, config_obj, app_obj):
        self.config_obj = config_obj
        self.app_obj = app_obj

    def run_experiment(self, test_start=10000000, test_end=11000000,
                       train_sizes=[100000], domain='twitter', start_fold=0,
                       clfs=['lr', 'rf', 'xgb'], metric='aupr'):
        """Configures the application based on the data subsets, and then runs
                the independent and relational models."""
        assert test_end > test_start
        assert start_fold >= 0

        rel_dir = self.config_obj.rel_dir
        out_dir = rel_dir + 'output/' + domain + '/learning_exp/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        ranges = self._create_ranges(test_start=test_start, test_end=test_end,
                                     train_sizes=train_sizes,
                                     start_fold=start_fold)
        print(ranges)

        rows = []
        cols = ['train_size']
        cols.extend(clfs)

        for start, end, train_pts, train_size, fold in ranges:
            row = [train_pts]

            for clf in clfs:
                d = self.app_obj.run(domain=domain, start=start, end=end,
                                     fold=fold, engine=None, clf=clf,
                                     ngrams=False, stacking=0, data='both',
                                     train_size=train_size, val_size=0,
                                     relations=[])

                row.append(d['ind'][metric])
            rows.append(row)

        self._write_scores_to_csv(rows, cols=cols, out_dir=out_dir,
                                  fname='ind_lrn.csv')

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

    def _create_ranges(self, test_start=100, test_end=200, train_sizes=[],
                       start_fold=0):
        test_size = test_end - test_start
        range_list = []

        for i, train_size in enumerate(train_sizes):
            tp = train_size / (train_size + test_size)
            start = test_start - train_size
            fold = str(start_fold + i)
            if start >= 0:
                range_list.append((start, test_end, train_size, tp, fold))
        return range_list

    def _write_scores_to_csv(self, rows, cols=[], out_dir='',
                             fname='results.csv'):
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(out_dir + fname, index=None)
