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
            start=0, end=1000, fold=0, data='both', ngrams=True,
            clf='lr', alter_user_ids=False, super_train=False,
            domain='twitter', separate_data=False, train_size=0.7,
            val_size=0.15, relations=['intext']):

        # validate args
        self.config_obj.set_options(domain=domain, start=start, end=end,
                                    train_size=train_size, val_size=val_size,
                                    ngrams=ngrams, clf=clf, engine=engine,
                                    fold=fold, relations=relations,
                                    stacking=stacking, data=data,
                                    separate_data=separate_data,
                                    alter_user_ids=alter_user_ids,
                                    super_train=super_train, modified=modified)

        # get data
        rels = self.config_obj.relations
        coms_df = self.data_obj.get_data(domain=domain, start=start, end=end)
        coms_df = self.data_obj.sep_data(coms_df, relations=relations,
                                         domain=domain, data=data)
        data = self.data_obj.split_data(coms_df, train_size=train_size,
                                        val_size=val_size)
        d = self._run_models(data, stacking=stacking, engine=engine)
        return d

        # if separate_data:
        #     relations = self.config_obj.relations
        #     no_rel_df, rel_df = self.data_obj.sep_rel_data(coms_df, relations,
        #                                                    domain=domain)

        #     print('\nNon-relational data...')
        #     data = self.data_obj.split_data(no_rel_df, train_size=train_size,
        #                                     val_size=val_size)
        #     dn = self._run_models(data, stacking=stacking, engine=None)
        #     score_dicts.append((dn, 'no_rel'))

        #     print('\nRelational data...')
        #     data = self.data_obj.split_data(rel_df, train_size=train_size,
        #                                     val_size=val_size)
        #     dr = self._run_models(data, stacking=stacking, engine=engine)
        #     score_dicts.append((dr, 'rel'))
        # else:
        #     print('\nRelational & Non-Relational data...')
        #     data = self.data_obj.split_data(coms_df, train_size=train_size,
        #                                     val_size=val_size)
        #     d = self._run_models(data, stacking=stacking, engine=engine)
        #     score_dicts.append((d, 'both'))

        return d

    # private
    def _run_psl(self, val_df, test_df):
        self.config_obj.engine = 'psl'
        self.config_obj.infer = False
        self.relational_obj.main(val_df, test_df)

        self.config_obj.infer = True
        self.relational_obj.main(val_df, test_df)

    def _run_mrf(self, val_df, test_df):
        self.config_obj.engine = 'mrf'
        self.relational_obj.main(val_df, test_df)

    def _run_models(self, data, stacking=0, engine='both'):
        print('running independent...')
        val_df, test_df = self.independent_obj.main(data)

        if engine is not None and (engine == 'psl' or engine == 'all'):
            print('running psl...')
            self.relational_obj.compile_reasoning_engine()
            self._run_psl(val_df, test_df)

        if engine is not None and (engine == 'mrf' or engine == 'all'):
            print('running mrf...')
            self._run_mrf(val_df, test_df)

        score_dict = self.analysis_obj.evaluate(test_df)
        print(score_dict)
        return score_dict
