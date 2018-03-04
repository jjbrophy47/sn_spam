"""
Module with high a high level API to run any part of the app.
"""
import os
import pandas as pd


class App:

    # public
    def __init__(config_obj, data_obj):
        self.config_obj = config_obj
        self.data_obj = data_obj

    def run(self, modified=False, pseudo=True, stacking=0, engine='both',
            start=0, end=1000, fold=0, separate_relations=True, ngrams=True,
            clf='lr', alter_user_ids=False, super_train=False,
            domain='twitter', separate_data=False):
        # validate args
        isinstance(modified, bool)
        isinstance(pseudo, bool)
        isinstance(separate_relations, bool)
        isinstance(ngrams, bool)
        isinstance(alter_user_ids, bool)
        isinstance(super_train, bool)
        isinstance(separate_models, bool)
        assert stacking >= 0
        assert engine in ['psl', 'mrf', 'both', None]
        assert clf in ['rf', 'lr']

        # get data
        coms_df = self.data_obj.get_data(domain, start, end)

        if seperate_data:
            non_rel_df, rel_df = data_obj.seperate_relational_data()
            self._run_models(non_rel_df, stacking=stacking, engine=engine)
            self._run_models(rel_df, stacking=stacking, engine=engine)
        else:
            self._run_models(coms_df, stacking=stacking, engine=engine)

        # score models
        score_dict = self.runner_obj.run_evaluation(test_df)
        return score_dict

    # private
    def _run_psl(self, val_df, test_df):
        self.config_obj.engine = 'psl'
        self.config.infer = False
        self.runner_obj.run_relational(val_df, test_df)

        self.config.infer = True
        self.runner_obj.run_relational(val_df, test_df)

    def _run_mrf(self, val_df, test_df):
        self.config.engine = 'mrf'
        self.runner_obj.run_relational(val_df, test_df)

    def _run_models(self, coms_df, stacking=0, engine='both'):
        val_df, test_df = self.runner_obj.run_independent(coms_df,
                stacking=stacking)
        if engine is not None and (engine='psl' or engine='both'):
            self._run_psl(val_df, test_df)
        if engine is not None and (engine='mrf' or engine='both'):
            self._run_mrf(val_df, test_df)
