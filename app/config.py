"""
Class that maintains the state of the app.
"""


class Config:

    def __init__(self):
        self.app_dir = None  # absolute path to app package.
        self.ind_dir = None  # absolute path to independent package.
        self.rel_dir = None  # absolute path to relational package.
        self.ana_dir = None  # absolute path to analysis package.
        self.fold = None  # experiment identifier.
        self.domain = None  # domain to model.
        self.start = None   # line number to start reading data.
        self.end = None   # line number to read data until.
        self.train_size = None  # amount of data to train independent mdoels.
        self.val_size = None  # amount of data to train relational models.
        self.ngrams = False  # switch to use ngrams as textual features.
        self.classifier = 'lr'  # independent classifier.
        self.relations = None  # relations to exploit.
        self.display = False  # has display if True, otherwise does not.
        self.modified = False  # data where msgs by a user are labeled same.
        self.infer = False  # switch to do inference for psl, otherwise train.
        self.alter_user_ids = False  # make all user ids in test set unique.
        self.super_train = True  # use train and val for training.
        self.evaluation = 'cc'  # cross-compare (cc) or train-test (tt).
        self.param_search = 'single'  # amount of hyper-parameters to search.
        self.tune_size = 0.15  # percentage of training data for tuning.
        self.engine = 'all'  # reasoning engine for collective classification.
        self.stacking = 0  # rounds to compute pseudo-relatonal features.
        self.data = 'both'  # controls which type of data to use.
        self.separate_relations = False  # disjoin training and test sets.

    # public
    def set_display(self, has_display):
        """If True, then application is run on a console.
        has_display: boolean indiciating if on a console."""
        self.display = has_display

    def set_directories(self, app_dir, ind_dir, rel_dir, ana_dir):
        """Sets absolute path directories in the config object.
        condig_dir: absolute path to the config package.
        ind_dir: absolute path to the independent package.
        rel_dir: absolute path to the relational package.
        ana_dir: absolute path to the analysis package."""
        self.app_dir = app_dir
        self.ind_dir = ind_dir
        self.rel_dir = rel_dir
        self.ana_dir = ana_dir

    def set_options(self, domain='twitter', start=0, end=1000,
                    train_size=0.7, val_size=0.15, ngrams=True, clf='lr',
                    engine='both', fold=0, relations=['intext'], stacking=0,
                    separate_relations=False, data='both',
                    alter_user_ids=False, super_train=False, modified=False,
                    evaluation='cc', param_search='single', tune_size=0.15):

        # validate args
        assert isinstance(ngrams, bool)
        assert isinstance(separate_relations, bool)
        assert isinstance(alter_user_ids, bool)
        assert isinstance(super_train, bool)
        assert isinstance(modified, bool)
        assert stacking >= 0
        assert evaluation in ['cc', 'tt']
        assert param_search in ['single', 'low', 'med', 'high']
        assert tune_size >= 0
        assert engine in self._available_engines()
        assert domain in self._available_domains()
        assert data in ['ind', 'rel', 'both']
        assert train_size + val_size < 1.0
        assert start < end
        assert clf in ['lr', 'rf', 'xgb']
        assert set(relations).issubset(self._available_relations()[domain])

        d = {'domain': domain, 'start': start, 'end': end,
             'train_size': train_size, 'val_size': val_size, 'ngrams': ngrams,
             'classifier': clf, 'engine': engine, 'fold': fold,
             'relations': relations, 'separate_relations': separate_relations,
             'data': data, 'alter_user_ids': alter_user_ids,
             'super_train': super_train, 'modified': modified,
             'stacking': stacking, 'evaluation': evaluation,
             'param_search': param_search, 'tune_size': tune_size}

        self._validate_config(d)
        self._populate_config(d)
        print(self)
        return self

    # private
    def _available_domains(self):
        return ['soundcloud', 'youtube', 'twitter', 'toxic',
                'ifwe', 'yelp_hotel', 'yelp_restaurant', 'adclicks']

    def _available_relations(self):
        relations = {}
        relations['soundcloud'] = ['posts', 'intext', 'intrack']
        relations['youtube'] = ['posts', 'intext', 'inment', 'inhour',
                                'invideo']
        relations['twitter'] = ['posts', 'intext', 'inhash', 'inment',
                                'inlink']
        relations['toxic'] = ['intext', 'inlink']
        relations['ifwe'] = ['inr0', 'inr1', 'inr2', 'inr3', 'inr4', 'inr5',
                             'inr6', 'inr7', 'insex', 'inage', 'intimepassed']
        relations['yelp_hotel'] = ['posts', 'intext', 'inhotel']
        relations['yelp_restaurant'] = ['posts', 'intext', 'inrest']
        relations['adclicks'] = ['hasip', 'inchannel', 'inapp', 'hasos',
                                 'hasdevice']
        return relations

    def _available_groups(self):
        groups = {'posts': 'user', 'intext': 'text', 'intrack': 'track',
                  'inhash': 'hash', 'inment': 'ment', 'invideo': 'video',
                  'inhour': 'hour', 'inlink': 'link', 'inhotel': 'hotel',
                  'inrest': 'rest', 'inr0': 'r0', 'inr1': 'r1', 'inr2': 'r2',
                  'inr3': 'r3', 'inr4': 'r4', 'inr5': 'r5', 'inr6': 'r6',
                  'inr7': 'r7', 'insex': 'sex', 'inage': 'age',
                  'intimepassed': 'timepassed', 'inlink': 'link',
                  'hasip': 'ip', 'inchannel': 'channel', 'inapp': 'app',
                  'hasos': 'os', 'hasdevice': 'device'}
        return groups

    def _available_ids(self):
        ids = {'posts': 'user_id', 'intext': 'text_id', 'intrack': 'track_id',
               'inhash': 'hashtag_id', 'inment': 'mention_id',
               'invideo': 'video_id', 'inhour': 'hour_id', 'inlink': 'link_id',
               'inhotel': 'hotel_id', 'inrest': 'rest_id', 'inr0': 'r0_id',
               'inr1': 'r1_id', 'inr2': 'r2_id', 'inr3': 'r3_id',
               'inr4': 'r4_id', 'inr5': 'r5_id', 'inr6': 'r6_id',
               'inr7': 'r7_id', 'insex': 'sex_id', 'inage': 'age_id',
               'intimepassed': 'time_passed_id', 'hasip': 'ip_id',
               'inchannel': 'channel_id', 'inapp': 'app_id', 'hasos': 'os_id',
               'hasdevice': 'device_id'}
        return ids

    def _groups_for_relations(self, relations):
        available_groups = self._available_groups()
        groups = [available_groups[relation] for relation in relations]
        return groups

    def _ids_for_relations(self, relations):
        available_ids = self._available_ids()
        ids = [available_ids[relation] for relation in relations]
        return ids

    def _available_engines(self):
        return ['psl', 'tuffy', 'mrf', 'all', None]

    def _populate_config(self, config):
        relations = config['relations']
        groups = self._groups_for_relations(relations)
        ids = self._ids_for_relations(relations)

        self.relations = list(zip(relations, groups, ids))
        self.domain = str(config['domain'])
        self.start = int(config['start'])
        self.end = int(config['end'])
        self.train_size = float(config['train_size'])
        self.val_size = float(config['val_size'])
        self.ngrams = bool(config['ngrams'])
        self.classifier = str(config['classifier'])
        self.fold = str(config['fold'])
        self.engine = str(config['engine'])
        self.stacking = int(config['stacking'])
        self.evaluation = str(config['evaluation'])
        self.evaluation = str(config['param_search'])
        self.evaluation = str(config['tune_size'])

    def __str__(self):
        relations = [r[0] for r in self.relations]

        s = '\nDomain: ' + str(self.domain) + '\n'
        s += 'Data range: ' + str(self.start) + ' to ' + str(self.end) + '\n'
        s += 'Training size: ' + str(self.train_size) + '\n'
        s += 'Validation size: ' + str(self.val_size) + '\n'
        s += 'Independent classifier: ' + str(self.classifier) + '\n'
        s += 'N-grams: ' + ('yes' if self.ngrams else 'no') + '\n'
        s += 'Stacks: ' + str(self.stacking) + '\n'
        s += 'Fold: ' + str(self.fold) + '\n'
        s += 'Relations: ' + str(relations) + '\n'
        s += 'Engine: ' + str(self.engine) + '\n'
        s += 'Evaluation: ' + str(self.evaluation) + '\n'
        s += 'Param search: ' + str(self.param_search) + '\n'
        s += 'Tuning size: ' + str(self.tune_size)
        return s
