"""
Module to test different test sets in a domain.
"""
import os
import pandas as pd


class Subsets_Experiment:

    def __init__(self, config_obj, app_obj):
        self.config_obj = config_obj
        self.app_obj = app_obj

    # public
    def divide_data(self, subsets=100, start=0, end=1000, fold=0):
        """Partitions the data into a specified number of subsets to run the
                independent and relational models on.
        num_subsets: number of partitions to split the data into.
        Returns a list of tuples containing the start and end of the data
                subset, as well as the experiment number."""
        data_size = end - start
        subset_size = data_size / subsets
        subsets = []

        for i in range(subsets):
            sub_start = int(start + (i * subset_size))
            sub_end = int(start + subset_size)
            sub_fold = str(int(fold) + i)
            subset = (sub_start, sub_end, sub_fold)
            subsets.append(subset)
        return subsets

    def run_experiment(self, start=0, end=1000, fold=0, domain='twitter',
                       subsets=100):
        """Configures the application based on the data subsets, and then runs
                the independent and relational models."""
        self._clear_data()

        rel_dir = self.config_obj.rel_dir
        out_dir = rel_dir + 'output/' + domain + '/subsets_exp/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        subsets = self.divide_data(start=start, end=end, fold=fold,
                                   subsets=subsets)

        rows = []
        for start, end, fold in subsets:
            d = self.app_obj.run(domain=domain, start=start, end=end,
                                 fold=fold, engine='all', clf='lr',
                                 ngrams=True, stacking=0, separate_data=True,
                                 alter_user_ids=False, super_train=False,
                                 train_size=0.7, val_size=0.15, modified=False,
                                 relations=['intext', 'posts', 'inment'],
                                 separate_relations=True)

            # TODO: handle case where d has more than 1 item.

            row = tuple()
            for model_name, scores in d.items():
                row += scores
            rows.append(row)

            columns = []
            for key, value in d.items():
                for i in range(len(value)):
                    column = key + '_' + str(i)
                    columns.append(column)

            df = pd.DataFrame(rows, columns=columns)
            df.to_csv(out_dir + 'results.csv', index=None)

    # private
    def _clear_data(self):
        ind_dir = self.config_obj.ind_dir
        rel_dir = self.config_obj.rel_dir
        domain = self.config_obj.domain

        fold_dir = ind_dir + '/data/' + domain + '/folds/'
        ind_pred_dir = ind_dir + '/output/' + domain + '/predictions/'
        rel_pred_dir = rel_dir + '/output/' + domain + '/predictions/'

        os.system('rm %s*.csv' % (fold_dir))
        os.system('rm %s*.csv' % (ind_pred_dir))
        os.system('rm %s*.csv' % (rel_pred_dir))
