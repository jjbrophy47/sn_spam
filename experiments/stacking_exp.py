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
                       clf='lr', start_stack=0, end_stack=7,
                       relations=[], fold=0, train_size=0.8, sim_dir=None):
        assert end_stack >= start_stack

        rel_dir = self.config_obj.rel_dir
        out_dir = rel_dir + 'output/' + domain + '/experiments/'
        self.util_obj.create_dirs(out_dir)

        fold = str(fold)
        fn = fold + '_stk.csv'

        rows = []
        cols = ['stacks', 'ind_aupr', 'ind_auroc']

        for i, stack in enumerate(range(start_stack, end_stack + 1)):
            for stack_splits in self._stack_splits(stack=stack):
                row = ['_'.join([str(stack), str(stack_splits)])]

                d = self.app_obj.run(domain=domain, start=start, end=end,
                                     fold=fold, engine=None, clf=clf,
                                     stacking=stack, data='both',
                                     train_size=train_size,
                                     val_size=0, relations=relations,
                                     sim_dir=sim_dir,
                                     stack_splits=stack_splits)

                for metric in ['aupr', 'auroc']:
                    row.append(d['ind'][metric])
                rows.append(row)

                self._write_scores_to_csv(rows, cols=cols, out_dir=out_dir,
                                          fname=fn)

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

    def _stack_splits(self, stack=0):
        ss = []

        splits = [0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875]

        if stack == 0:
            ss.append([])

        if stack == 1:
            for split in splits:
                ss.append([split])

        elif stack == 2:
            for split1 in splits:
                for split2 in splits:
                    ss.append([split1, split2])

        return ss

    def _write_scores_to_csv(self, rows, cols=[], out_dir='',
                             fname='results.csv'):
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(out_dir + fname, index=None)
