"""
Module to test different test sets in a domain.
"""
import os
import pandas as pd
from collections import defaultdict


class Subsets_Experiment:

    # public
    def __init__(self, config_obj, app_obj):
        self.config_obj = config_obj
        self.app_obj = app_obj

    def run_experiment(self, start=0, end=1000, fold=0, domain='twitter',
                       subsets=100):
        """Configures the application based on the data subsets, and then runs
                the independent and relational models."""
        self._clear_data(domain=domain)

        rel_dir = self.config_obj.rel_dir
        out_dir = rel_dir + 'output/' + domain + '/subsets_exp/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        subsets = self._divide_data(start=start, end=end, fold=fold,
                                    subsets=subsets)

        rows = defaultdict(lambda: [])
        cols = defaultdict(lambda: [])
        names = set()
        for start, end, fold in subsets:
            sc = self.app_obj.run(domain=domain, start=start, end=end,
                                  fold=fold, engine='all', clf='lr',
                                  ngrams=True, stacking=0, separate_data=False,
                                  alter_user_ids=False, super_train=True,
                                  train_size=0.7, val_size=0.15,
                                  modified=False,
                                  relations=['intext', 'posts', 'inment'],
                                  separate_relations=True)

            for d, name in sc:
                row = []
                for model_name, sd in d.items():
                    row.extend(sd.values())
                rows[name].append(row)

                if cols[name] == []:
                    for model, v in d.items():
                        for score in v.keys():
                            cols[name].append(model + '_' + score)
                names.add(name)

        for name in names:
            self._write_scores_to_csv(rows[name], cols=cols[name],
                                      out_dir=out_dir, fname=name + '_res.csv')

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
