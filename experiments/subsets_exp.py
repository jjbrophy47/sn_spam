"""
Module to test different test sets in a domain.
"""
import os
import pandas as pd


class Subsets_Experiment:

    # public
    def __init__(self, config_obj, app_obj, util_obj):
        self.config_obj = config_obj
        self.app_obj = app_obj
        self.util_obj = util_obj

    def run_experiment(self, start=0, end=1000, fold=0, domain='twitter',
                       subsets=100, data='both', train_size=0.8,
                       val_size=0.1, relations=[], clf='lgb',
                       engine='all', featuresets=['all'],
                       stacking=0):
        rel_dir = self.config_obj.rel_dir
        out_dir = rel_dir + 'output/' + domain + '/experiments/'
        self.util_obj.create_dirs(out_dir)

        subsets = self._divide_data(start=start, end=end, fold=fold,
                                    subsets=subsets)

        rows, cols = [], []
        for start, end, fold in subsets:
            d = self.app_obj.run(domain=domain, start=start, end=end,
                                 fold=fold, engine=engine, clf=clf,
                                 stacking=stacking, data=data,
                                 featuresets=featuresets, relations=relations,
                                 train_size=train_size, val_size=val_size)

            row = []
            for model_name, sd in d.items():
                row.extend(sd.values())
            rows.append(row)

            if cols == []:
                for model, v in d.items():
                    for score in v.keys():
                        cols.append(model + '_' + score)

        fn = data + '_subsets.csv'
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

    def _divide_data(self, subsets=100, start=0, end=1000, fold=0):
        data_size = end - start
        subset_size = data_size / subsets
        subsets_list = []

        for i in range(subsets):
            sub_start = int(start + (i * subset_size))
            sub_end = int(sub_start + subset_size)
            sub_fold = str(int(fold) + i)
            subset = (sub_start, sub_end, sub_fold)
            subsets_list.append(subset)
        return subsets_list

    def _write_scores_to_csv(self, rows, cols=[], out_dir='',
                             fname='results.csv'):
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(out_dir + fname, index=None)
