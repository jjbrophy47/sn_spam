"""
Class that maintains the state of the app.
"""


class Config:

    def __init__(self):
        self.domain = None
        """Social network to work with."""
        self.start = None
        """Beginning of comment subset."""
        self.end = None
        """Ending of comment subset."""
        self.train_size = None
        """Amount of data to be used for training."""
        self.val_size = None
        """Amount of data to be used for relational model training."""
        self.data = 'both'
        """Whether to use independent data, relational data, or both."""
        self.fold = None
        """Identifier for outputting files."""
        self.ngrams = False
        """Use of ngrams in independent model."""
        self.stacking = 0
        """Use pseudo-relatonal features in independent model."""
        self.classifier = None
        """Classifier to use in independent model."""
        self.relations = None
        """Relations to exploit in relational model."""
        self.engine = None
        """Reasoning engine for inference."""
        self.config_dir = None
        """Absolute path to the config package."""
        self.ind_dir = None
        """Absolute path to the independent package."""
        self.rel_dir = None
        """Absolute path to the relational package."""
        self.ana_dir = None
        """Absolute path to the analysis package."""
        self.display = False
        """Boolean indicating whether application is running on a console."""
        self.modified = False
        """Boolean to indicate which comments file to use."""
        self.infer = False
        """Boolean to train relational model if False, otherwise infer."""
        self.alter_user_ids = False
        """Boolean to alter user ids if doing robustness testing."""
        self.super_train = False
        """Boolean to use both train and val for training if True."""
        self.separate_relations = False
        """Boolean to disjoin relations between training and test sets."""

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
                    alter_user_ids=False, super_train=False, modified=False):
        assert isinstance(ngrams, bool)
        assert isinstance(separate_relations, bool)
        assert isinstance(alter_user_ids, bool)
        assert isinstance(super_train, bool)
        assert isinstance(modified, bool)
        assert stacking >= 0

        d = {'domain': domain, 'start': start, 'end': end,
             'train_size': train_size, 'val_size': val_size, 'ngrams': ngrams,
             'classifier': clf, 'engine': engine, 'fold': fold,
             'relations': relations, 'separate_relations': separate_relations,
             'data': data, 'alter_user_ids': alter_user_ids,
             'super_train': super_train, 'modified': modified,
             'stacking': stacking}

        self._validate_config(d)
        self._populate_config(d)
        print(self)
        return self

    # def parse_config(self):
    #     """Returns a config object after reading and parsing a config file."""
    #     items = self.parsable_items()
    #     config = self.read_config_file(self.app_dir + 'config.txt', items)
    #     self.validate_config(config)
    #     self.populate_config(config)
    #     print(self)
    #     return self

    # # private
    # def parsable_items(self):
    #     """List of items in the config file to parse."""
    #     items = ['domain', 'start', 'end', 'train_size', 'val_size',
    #              'classifier', 'ngrams', 'pseudo', 'fold',
    #              'relations', 'engine']
    #     return items

    # def read_config_file(self, filename, items):
    #     """Reads config file and parses each line.
    #     filename: name of the config file.
    #     items: list of parsable items.
    #     Returns dict full of config keys and values."""
    #     line_num, config = 0, {}

    #     with open(filename) as f:
    #         for line in f:
    #             config = self.parse_line(line.strip(), line_num, config, items)
    #             line_num += 1
    #     return config

    # def parse_line(self, line, line_num, config, items):
    #     """Parses a line in the config file and adds it to the dictionary.
    #     line: string to parse.
    #     line_num: index number of the line.
    #     config: dict of config values.
    #     items: list of parsable items.
    #     Returns dict filled with config key value pairs."""
    #     start = line.find('=')
    #     if start != -1:
    #         end = line.find(' ')
    #         config[items[line_num]] = line[start + 1:end]

    #         if items[line_num] == 'relations':
    #             end = line.find(']')
    #             config[items[line_num]] = line[start + 1:end + 1]
    #             relations = config['relations']
    #             config['relations'] = relations[1:len(relations) - 1].split(',')
    #             config['relations'] = [x.strip() for x in config['relations']]
    #     return config

    # private
    def _available_domains(self):
        return ['soundcloud', 'youtube', 'twitter', 'toxic',
                'ifwe', 'yelp_hotel', 'yelp_restaurant']

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
        return relations

    def _available_groups(self):
        groups = {'posts': 'user', 'intext': 'text', 'intrack': 'track',
                  'inhash': 'hash', 'inment': 'ment', 'invideo': 'video',
                  'inhour': 'hour', 'inlink': 'link', 'inhotel': 'hotel',
                  'inrest': 'rest', 'inr0': 'r0', 'inr1': 'r1', 'inr2': 'r2',
                  'inr3': 'r3', 'inr4': 'r4', 'inr5': 'r5', 'inr6': 'r6',
                  'inr7': 'r7', 'insex': 'sex', 'inage': 'age',
                  'intimepassed': 'timepassed', 'inlink': 'link'}
        return groups

    def _available_ids(self):
        ids = {'posts': 'user_id', 'intext': 'text_id', 'intrack': 'track_id',
               'inhash': 'hash_id', 'inment': 'ment_id', 'invideo': 'vid_id',
               'inhour': 'hour_id', 'inlink': 'link_id',
               'inhotel': 'hotel_id', 'inrest': 'rest_id', 'inr0': 'r0_id',
               'inr1': 'r1_id', 'inr2': 'r2_id', 'inr3': 'r3_id',
               'inr4': 'r4_id', 'inr5': 'r5_id', 'inr6': 'r6_id',
               'inr7': 'r7_id', 'insex': 'sex_id', 'inage': 'age_id',
               'intimepassed': 'time_passed_id', 'inlink': 'link_id'}
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

    def _validate_config(self, config):
        """Makes sure that the domain is valid and that the chosen relations
        are valid for that domain.
        config: dict object with config values."""
        relations = self._available_relations()

        if not config['domain'] in self._available_domains():
            print('domain ' + str(config['domain']) + ' invalid, exiting...')
            exit(0)

        if not set(config['relations']).issubset(relations[config['domain']]):
            s = 'relations ' + str(config['relations']) + ' invalid '
            s += 'for this domain, exiting...'
            print(s)
            exit(0)

        if not config['engine'] in self._available_engines():
            print('engine ' + str(config['engine']) + ' invalid, exiting...')
            exit(0)

        if int(config['start']) > int(config['end']):
            print('start must come before end, exiting...')
            exit(0)

        data = float(config['train_size']) + float(config['val_size'])
        if data >= 1.0:
            print('train and val must add up to less than 1.0, exiting...')
            exit(0)

        if config['classifier'] not in ['lr', 'rf']:
            print('available classifiers: {lr, rf}, exiting...')
            exit(0)

        if config['data'] not in ['ind', 'rel', 'both']:
            print('data must be: {ind, rel, or both}, exiting...')
            exit(0)

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

    def __str__(self):
        relations = [r[0] for r in self.relations]

        s = 'Domain: ' + str(self.domain) + '\n'
        s += 'Data range: ' + str(self.start) + ' to ' + str(self.end) + '\n'
        s += 'Training size: ' + str(self.train_size) + '\n'
        s += 'Validation size: ' + str(self.val_size) + '\n'
        s += 'Independent classifier: ' + str(self.classifier) + '\n'
        s += 'N-grams: ' + ('yes' if self.ngrams else 'no') + '\n'
        s += 'Stacks: ' + str(self.stacking) + '\n'
        s += 'Fold: ' + str(self.fold) + '\n'
        s += 'Relations: ' + str(relations) + '\n'
        s += 'Engine: ' + str(self.engine)
        return s
