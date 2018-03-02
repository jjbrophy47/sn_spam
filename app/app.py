"""
Module with high a high level API to run any part of the app.
"""
import os
import pandas as pd


class App:

    # public
    def __init__(config_obj):
        self.config_obj = config_obj

    def run(self, modified=False, pseudo=True, stacking=0, engine='both',
            start=0, end=1000, fold=0, separate_relations=True, ngrams=True,
            clf='lr', alter_user_ids=False, super_train=False,
            domain='twitter'):
        # validate args
        isinstance(modified, bool)
        isinstance(pseudo, bool)
        isinstance(separate_relations, bool)
        isinstance(ngrams, bool)
        isinstance(alter_user_ids, bool)
        isinstance(super_train, bool)
        assert stacking >= 0
        assert engine in ['psl', 'mrf', 'both', None]
        assert clf in ['rf', 'lr']

        # run models
        val_df, test_df = self.runner_obj.run_independent(stacking=stacking)

        if engine is not None and (engine='psl' or engine='both'):
            self.run_psl(val_df, test_df)

        if engine is not None and (engine='mrf' or engine='both'):
            self.run_mrf(val_df, test_df)

        # score models
        score_dict = self.runner_obj.run_evaluation(test_df)
        return score_dict

    def run_psl(self, val_df, test_df):
        self.config_obj.engine = 'psl'
        self.config.infer = False
        self.runner_obj.run_relational(val_df, test_df)

        self.config.infer = True
        self.runner_obj.run_relational(val_df, test_df)

    def run_mrf(self, val_df, test_df):
        self.config.engine = 'mrf'
        self.runner_obj.run_relational(val_df, test_df)
