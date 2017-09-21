"""
Module to handle the necessary configuration file.
"""


class Config:
    """Parses the config file and returns a config object."""

    def __init__(self):
        """Inits Config object with default values."""

        self.domain = None
        """Social network to work with."""
        self.start = None
        """Beginning of comment subset."""
        self.end = None
        """Ending of comment subset."""
        self.train_size = None
        """Amount of data to be used for training."""
        self.fold = None
        """Identifier for outputting files."""
        self.ngrams = False
        """Use of ngrams in independent model."""
        self.pseudo = False
        """Use relatonal features in independent model."""
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
        self.debug = False
        """Boolean to output debugging information."""
        self.modified = False
        """Boolean to indicate which comments file to use."""
        self.saved = False
        """Boolean to use pre-trained models if True, otherwise retrain."""

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

    def set_options(self, args):
        """Sets additional configuration options from the commandline.
        args: list of flags that turns on various options."""
        # print(args)
        if '-m' in args:
            self.modified = True
        if '-d' in args:
            self.debug = True
        if '-s' in args:
            self.saved = True

    def parse_config(self):
        """Returns a config object after reading and parsing a config file."""
        items = self.parsable_items()
        config = self.read_config_file(self.app_dir + 'config.txt', items)
        self.validate_config(config)
        self.populate_config(config)
        print(self)
        return self

    # private
    def parsable_items(self):
        """List of items in the config file to parse."""
        items = ['domain', 'start', 'end', 'train_size', 'classifier',
                 'ngrams', 'pseudo', 'fold', 'relations', 'engine',
                 'model', 'debug']
        return items

    def read_config_file(self, filename, items):
        """Reads config file and parses each line.
        filename: name of the config file.
        items: list of parsable items.
        Returns dict full of config keys and values."""
        line_num, config = 0, {}

        with open(filename) as f:
            for line in f:
                config = self.parse_line(line.strip(), line_num, config, items)
                line_num += 1
        return config

    def parse_line(self, line, line_num, config, items):
        """Parses a line in the config file and adds it to the dictionary.
        line: string to parse.
        line_num: index number of the line.
        config: dict of config values.
        items: list of parsable items.
        Returns dict filled with config key value pairs."""
        start = line.find('=')
        if start != -1:
            end = line.find(' ')
            config[items[line_num]] = line[start + 1:end]

            if items[line_num] == 'relations':
                end = line.find(']')
                config[items[line_num]] = line[start + 1:end + 1]
                relations = config['relations']
                config['relations'] = relations[1:len(relations) - 1].split(',')
                config['relations'] = [x.strip() for x in config['relations']]
        return config

    def available_domains(self):
        """Social networks available to use as data."""
        return ['soundcloud', 'youtube', 'twitter', 'yelp_hotel',
                'yelp_restaurant']

    def available_relations(self):
        """Returns a dictionary specifiying which relations are available
        for each domain."""
        relations = {}
        relations['soundcloud'] = ['posts', 'intext', 'intrack']
        relations['youtube'] = ['posts', 'intext', 'inment', 'inhour',
                'invideo']
        relations['twitter'] = ['posts', 'intext', 'inhash', 'inment',
                'inlink']
        relations['yelp_hotel'] = ['posts', 'intext', 'inhotel']
        relations['yelp_restaurant'] = ['posts', 'intext', 'inrest']
        return relations

    def available_groups(self):
        """Returns a dictionary mapping a relation to a more general term
        for a group of that relation."""
        groups = {'posts': 'user', 'intext': 'text', 'intrack': 'track',
                'inhash': 'hash', 'inment': 'ment', 'invideo': 'video',
                'inhour': 'hour', 'inlink': 'link', 'inhotel': 'hotel',
                'inrest': 'rest'}
        return groups

    def available_ids(self):
        """Returns a dictionary mapping a relation to the column name used to
        group data of that relation together."""
        ids = {'posts': 'user_id', 'intext': 'text_id', 'intrack': 'track_id',
                'inhash': 'hash_id', 'inment': 'ment_id', 'invideo': 'vid_id',
                'inhour': 'hour_id', 'inlink': 'link_id',
                'inhotel': 'hotel_id', 'inrest': 'rest_id'}
        return ids

    def groups_for_relations(self, relations):
        """Maps a list of relations to a list of groups for those relations.
        relations: list of relations to map (e.g. intext, posts, etc.).
        Returns a list of corresponding group terms for those relations."""
        available_groups = self.available_groups()
        groups = [available_groups[relation] for relation in relations]
        return groups

    def ids_for_relations(self, relations):
        """Maps a list of relations to a list of group ids for those relations.
        relations: list of relations to map (e.g. intext, posts, etc.).
        Returns a list of corresponding group ids for those relations."""
        available_ids = self.available_ids()
        ids = [available_ids[relation] for relation in relations]
        return ids

    def available_engines(self):
        """Returns list of relational engines to use."""
        return ['psl', 'tuffy']

    def validate_config(self, config):
        """Makes sure that the domain is valid and that the chosen relations
        are valid for that domain.
        config: dict object with config values."""
        relations = self.available_relations()

        if not config['domain'] in self.available_domains():
            print('domain ' + str(config['domain']) + ' invalid, exiting...')
            exit(0)

        if not set(config['relations']).issubset(relations[config['domain']]):
            s = 'relations ' + str(config['relations']) + ' invalid '
            s += 'for this domain, exiting...'
            print(s)
            exit(0)

        if not config['engine'] in self.available_engines():
            print('engine ' + str(config['engine']) + ' invalid, exiting...')
            exit(0)

        if int(config['start']) > int(config['end']):
            print('start must come before end, exiting...')
            exit(0)

    def populate_config(self, config):
        """Populates the object with the config dictionary.
        config: dict of values to populate config object with."""
        relations = config['relations']
        groups = self.groups_for_relations(relations)
        ids = self.ids_for_relations(relations)

        self.relations = list(zip(relations, groups, ids))
        self.domain = str(config['domain'])
        self.start = int(config['start'])
        self.end = int(config['end'])
        self.train_size = float(config['train_size'])
        self.ngrams = True if config['ngrams'].lower() == 'yes' else False
        self.pseudo = True if config['pseudo'].lower() == 'yes' else False
        self.classifier = str(config['classifier'])
        self.fold = str(config['fold'])
        self.engine = str(config['engine'])

    def __str__(self):
        """Inherent method to print out the config object."""
        classifier = 'logistic regression'
        if self.classifier == 'rf':
            classifier = 'random forest'
        relations = [r[0] for r in self.relations]

        s = 'Domain: ' + str(self.domain) + '\n'
        s += 'Data range: ' + str(self.start) + ' to ' + str(self.end) + '\n'
        s += 'Training size: ' + str(self.train_size) + '\n'
        s += 'Independent classifier: ' + str(classifier) + '\n'
        s += 'N-grams: ' + ('yes' if self.ngrams else 'no') + '\n'
        s += 'Use pseudo: ' + ('yes' if self.pseudo else 'no') + '\n'
        s += 'Fold: ' + str(self.fold) + '\n'
        s += 'Relations to exploit: ' + str(relations) + '\n'
        s += 'Engine: ' + str(self.engine) + '\n'
        s += 'Debug: ' + ('yes' if self.debug else 'no') + '\n'
        s += 'Use modified: ' + ('yes' if self.modified else 'no') + '\n'
        s += 'Use pre-trained: ' + ('yes' if self.saved else 'no')
        return s
