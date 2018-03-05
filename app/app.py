"""
Module with high a high level API to run any part of the app.
"""


class App:

    # public
    def __init__(self, config_obj, data_obj, independent_obj, relational_obj,
                 analysis_obj):
        self.config_obj = config_obj
        self.data_obj = data_obj
        self.independent_obj = independent_obj
        self.relational_obj = relational_obj
        self.analysis_obj = analysis_obj

    def run(self, modified=False, stacking=0, engine='all',
            start=0, end=1000, fold=0, separate_relations=True, ngrams=True,
            clf='lr', alter_user_ids=False, super_train=False,
            domain='twitter', separate_data=False, train_size=0.7,
            val_size=0.15, relations=['intext']):

        # validate args
        self.config_obj.set_options(domain=domain, start=start, end=end,
                                    train_size=train_size, val_size=val_size,
                                    ngrams=ngrams, clf=clf, engine=engine,
                                    fold=fold, relations=relations,
                                    separate_relations=separate_relations,
                                    separate_data=separate_data,
                                    alter_user_ids=alter_user_ids,
                                    super_train=super_train, modified=modified)

        # get data
        coms_df = self.data_obj.get_data(domain=domain, start=start, end=end)
        data = self.data_obj.split_data(coms_df, train_size=train_size,
                                        val_size=val_size)

        if separate_data:
            # TODO handle data dict.
            non_rel_df, rel_df = self.data_obj.sep_relational_data(coms_df)
            self._run_models(non_rel_df, stacking=stacking, engine=engine)
            self._run_models(rel_df, stacking=stacking, engine=engine)
        else:
            self._run_models(data, stacking=stacking, engine=engine)

    # private
    def _run_psl(self, val_df, test_df):
        print('running psl...')
        self.config_obj.engine = 'psl'
        self.config_obj.infer = False
        self.relational_obj.main(val_df, test_df)

        self.config_obj.infer = True
        self.relational_obj.main(val_df, test_df)

    def _run_mrf(self, val_df, test_df):
        print('running mrf...')
        self.config_obj.engine = 'mrf'
        self.relational_obj.main(val_df, test_df)

    def _run_models(self, data, stacking=0, engine='both'):
        print('running independent...')
        val_df, test_df = self.independent_obj.main(data)

        if engine is not None and (engine == 'psl' or engine == 'all'):
            self.relational_obj.compile_reasoning_engine()
            self._run_psl(val_df, test_df)

        if engine is not None and (engine == 'mrf' or engine == 'all'):
            self._run_mrf(val_df, test_df)

        score_dict = self.analysis_obj.evaluate(test_df)
        print(score_dict)
