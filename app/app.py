"""
Module with high a high level API to run any part of the app.
"""


class App:

    # public
    def __init__(self, config_obj, data_obj, independent_obj, relational_obj,
                 analysis_obj, util_obj):
        self.config_obj = config_obj
        self.data_obj = data_obj
        self.independent_obj = independent_obj
        self.relational_obj = relational_obj
        self.analysis_obj = analysis_obj
        self.util_obj = util_obj

    def run(self, modified=False, stacking=0, engine='all',
            start=0, end=1000, fold=0, data='both', ngrams=True,
            clf='lr', alter_user_ids=False, super_train=True,
            domain='twitter', separate_relations=True, train_size=0.7,
            val_size=0.15, relations=['intext'], evaluation='cc',
            param_search='single', tune_size=0.15, featureset='all'):

        # validate args
        self.config_obj.set_options(domain=domain, start=start, end=end,
                                    train_size=train_size, val_size=val_size,
                                    ngrams=ngrams, clf=clf, engine=engine,
                                    fold=fold, relations=relations,
                                    stacking=stacking, data=data,
                                    separate_relations=separate_relations,
                                    alter_user_ids=alter_user_ids,
                                    super_train=super_train, modified=modified,
                                    evaluation=evaluation, tune_size=tune_size,
                                    param_search=param_search,
                                    featureset=featureset)

        # get data
        if evaluation == 'cc':
            relations = self.config_obj.relations
            coms_df = self.data_obj.get_data(domain=domain, start=start,
                                             end=end, evaluation=evaluation)
            coms_df = self.data_obj.get_rel_ids(coms_df, domain, relations)
            coms_df = self.data_obj.sep_data(coms_df, relations=relations,
                                             domain=domain, data=data)
            dfs = self.data_obj.split_data(coms_df, train_size=train_size,
                                           val_size=val_size)
        elif evaluation == 'tt':
            train_df, test_df = self.data_obj.get_data(domain=domain,
                                                       start=start, end=end,
                                                       evaluation=evaluation)
            train_df = self.data_obj.get_rel_ids(train_df, domain, relations)
            test_df = self.data_obj.get_rel_ids(test_df, domain, relations)
            dfs = self.data_obj.split_data(train_df, train_size=train_size,
                                           val_size=None)
            dfs['test'] = test_df

        d = self._run_models(dfs, stacking=stacking, engine=engine, data=data,
                             evaluation=evaluation)
        return d

    # private
    def _run_models(self, dfs, stacking=0, engine='all', data='both',
                    evaluation='cc'):
        score_dict = None

        self.util_obj.out('running independent...')
        val_df, test_df = self.independent_obj.main(dfs)

        if data in ['rel', 'both'] and engine in ['psl', 'all']:
            self.util_obj.out('running psl...')
            self.relational_obj.compile_reasoning_engine()
            self._run_psl(val_df, test_df)

        if data in ['rel', 'both'] and engine in ['mrf', 'all']:
            self.util_obj.out('running mrf...')
            self._run_mrf(val_df, test_df)

        if evaluation == 'cc':
            score_dict = self.analysis_obj.evaluate(test_df)
            self.util_obj.out(str(score_dict) + '\n')

        return score_dict

    def _run_mrf(self, val_df, test_df):
        self.config_obj.engine = 'mrf'
        self.relational_obj.main(val_df, test_df)

    def _run_psl(self, val_df, test_df):
        self.config_obj.engine = 'psl'
        self.config_obj.infer = False
        self.relational_obj.main(val_df, test_df)

        self.config_obj.infer = True
        self.relational_obj.main(val_df, test_df)
