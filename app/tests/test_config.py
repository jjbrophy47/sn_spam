"""
Tests the config script.
"""
import unittest
import mock
from context import config


class ConfigTestCase(unittest.TestCase):
    def setUp(self):
        self.test_obj = config.Config()

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        # setup
        test_obj = self.test_obj

        # assert
        self.assertIsNone(test_obj.domain)
        self.assertIsNone(test_obj.start)
        self.assertIsNone(test_obj.end)
        self.assertIsNone(test_obj.train_size)
        self.assertIsNone(test_obj.fold)
        self.assertIsNone(test_obj.classifier)
        self.assertIsNone(test_obj.relations)
        self.assertIsNone(test_obj.engine)
        self.assertIsNone(test_obj.config_dir)
        self.assertIsNone(test_obj.ind_dir)
        self.assertIsNone(test_obj.rel_dir)
        self.assertIsNone(test_obj.ana_dir)
        self.assertFalse(test_obj.ngrams)
        self.assertFalse(test_obj.display)
        self.assertFalse(test_obj.debug)
        self.assertFalse(test_obj.modified)
        self.assertFalse(test_obj.saved)

    def test_parsable_items(self):
        # test
        result = self.test_obj.parsable_items()

        # assert
        self.assertTrue(len(result), 9)
        self.assertTrue('domain' in result)
        self.assertTrue('start' in result)
        self.assertTrue('end' in result)
        self.assertTrue('train_size' in result)
        self.assertTrue('ngrams' in result)
        self.assertTrue('classifier' in result)
        self.assertTrue('fold' in result)
        self.assertTrue('relations' in result)
        self.assertTrue('engine' in result)

    def test_read_config_file(self):
        test_config_file = 'app/tests/resources/sample_config.txt'
        items = self.test_obj.parsable_items()

        result = self.test_obj.read_config_file(test_config_file, items)

        self.assertTrue(result['domain'] == 'soundcloud')
        self.assertTrue(result['start'] == '0')
        self.assertTrue(result['end'] == '1000')
        self.assertTrue(result['train_size'] == '0.7')
        self.assertTrue(result['classifier'] == 'lr')
        self.assertTrue(result['fold'] == '32')
        self.assertTrue(result['relations'] == ['intext', 'posts', 'intrack'])
        self.assertTrue(result['ngrams'] == 'yes')
        self.assertTrue(result['engine'] == 'psl')

    def test_parse_line(self):
        line = 'train_size=0.7  # percentage of data to use for training.'
        line_num = 3
        config = {}
        items = self.test_obj.parsable_items()

        result = self.test_obj.parse_line(line, line_num, config, items)
        self.assertTrue(result['train_size'] == '0.7')

    def test_parse_line_relations(self):
        line = 'relations=[text,posts,hashtags]  # relations to exploit'
        line_num = 8
        config = {}
        items = self.test_obj.parsable_items()

        result = self.test_obj.parse_line(line, line_num, config, items)

        self.assertTrue(result['relations'] == ['text', 'posts', 'hashtags'])

    def test_available_domains(self):
        result = self.test_obj.available_domains()

        self.assertTrue(result == ['soundcloud', 'youtube', 'twitter'])

    def test_available_relations(self):
        sc_relations = ['posts', 'intext', 'intrack']
        yt_relations = ['posts', 'intext', 'inment', 'inhour', 'invideo']
        tw_relations = ['posts', 'intext', 'inhash', 'inment', 'inlink']

        result = self.test_obj.available_relations()

        self.assertTrue(result['soundcloud'] == sc_relations)
        self.assertTrue(result['youtube'] == yt_relations)
        self.assertTrue(result['twitter'] == tw_relations)

    def test_available_groups(self):
        result = self.test_obj.available_groups()

        self.assertTrue(len(result) == 8)
        self.assertTrue(result['posts'] == 'user')
        self.assertTrue(result['intext'] == 'text')
        self.assertTrue(result['inhash'] == 'hash')
        self.assertTrue(result['intrack'] == 'track')
        self.assertTrue(result['inlink'] == 'link')

    def test_available_ids(self):
        result = self.test_obj.available_ids()

        self.assertTrue(len(result) == 8)
        self.assertTrue(result['posts'] == 'user_id')
        self.assertTrue(result['intext'] == 'text_id')
        self.assertTrue(result['inhash'] == 'hash_id')
        self.assertTrue(result['intrack'] == 'track_id')
        self.assertTrue(result['inlink'] == 'link_id')

    def test_groups_for_relations(self):
        result = self.test_obj.groups_for_relations(['intext', 'posts'])

        self.assertTrue(result == ['text', 'user'])

    def test_ids_for_relations(self):
        result = self.test_obj.ids_for_relations(['intext', 'posts'])

        self.assertTrue(result == ['text_id', 'user_id'])

    def test_available_engines(self):
        result = self.test_obj.available_engines()

        self.assertTrue(result == ['psl', 'tuffy'])

    def test_validate_config(self):
        config = {'domain': 'soundcloud', 'relations': ['intext', 'posts'],
                  'ngrams': 'no', 'engine': 'psl', 'model': 'basic',
                  'start': '27', 'end': '77'}

        self.test_obj.validate_config(config)

    def test_validate_config_wrong_relation(self):
        config = {'domain': 'soundcloud', 'relations': ['text', 'fake'],
                  'ngrams': 'no', 'engine': 'psl'}

        with self.assertRaises(SystemExit) as cm:
            self.test_obj.validate_config(config)
        self.assertEqual(cm.exception.code, 0)

    def test_validate_config_wrong_domain(self):
        config = {'domain': 'fake', 'relations': ['text', 'fake']}

        with self.assertRaises(SystemExit) as cm:
            self.test_obj.validate_config(config)
        self.assertEqual(cm.exception.code, 0)

    def test_validate_config_wrong_engine(self):
        config = {'domain': 'soundcloud', 'relations': ['text', 'posts'],
                  'ngrams': 'no', 'engine': None}

        with self.assertRaises(SystemExit) as cm:
            self.test_obj.validate_config(config)
        self.assertEqual(cm.exception.code, 0)

    def test_validate_config_wrong_start_end(self):
        config = {'domain': 'soundcloud', 'relations': ['text', 'posts'],
                  'ngrams': 'no', 'engine': 'psl', 'start': '69',
                  'end': '27'}

        with self.assertRaises(SystemExit) as cm:
            self.test_obj.validate_config(config)
        self.assertEqual(cm.exception.code, 0)

    def test_populate_config(self):
        config = {'domain': 'soundcloud', 'start': '0', 'end': '69',
                  'train_size': '0.7', 'classifier': 'lr', 'fold': '1',
                  'relations': ['intext', 'posts'], 'model': 'basic',
                  'ngrams': 'no', 'engine': 'psl', 'debug': 'yes',
                  'pseudo': 'yes'}

        self.test_obj.populate_config(config)

        test_obj = self.test_obj
        expected = [('intext', 'text', 'text_id'),
                ('posts', 'user', 'user_id')]
        self.assertTrue(test_obj.domain == 'soundcloud')
        self.assertTrue(test_obj.start == 0)
        self.assertTrue(test_obj.end == 69)
        self.assertTrue(test_obj.train_size == 0.7)
        self.assertTrue(test_obj.classifier == 'lr')
        self.assertTrue(not test_obj.ngrams)
        self.assertTrue(test_obj.pseudo)
        self.assertTrue(test_obj.fold == '1')
        self.assertTrue(test_obj.relations == expected)
        self.assertTrue(test_obj.engine == 'psl')

    def test_set_display(self):
        self.test_obj.set_display(True)

        self.assertTrue(self.test_obj.display)

    def test_set_directories(self):
        self.test_obj.set_directories('a/', 'b/', 'c/', 'd/')

        self.assertTrue(self.test_obj.app_dir == 'a/')
        self.assertTrue(self.test_obj.ind_dir == 'b/')
        self.assertTrue(self.test_obj.rel_dir == 'c/')
        self.assertTrue(self.test_obj.ana_dir == 'd/')

    def test_set_options_modified(self):
        self.test_obj.set_options(['-m', '-e'])

        self.assertTrue(self.test_obj.modified)
        self.assertTrue(not self.test_obj.debug)

    def test_set_options_all(self):
        self.test_obj.set_options(['-m', '-d', '-e', '-s'])

        self.assertTrue(self.test_obj.modified)
        self.assertTrue(self.test_obj.debug)
        self.assertTrue(self.test_obj.saved)

    def test_parse_config(self):
        self.test_obj.parsable_items = mock.Mock(return_value='items')
        self.test_obj.read_config_file = mock.Mock(return_value='config')
        self.test_obj.validate_config = mock.Mock()
        self.test_obj.populate_config = mock.Mock()
        self.test_obj.relations = [('intext', 'text', 'text_id')]
        self.test_obj.app_dir = 'c/'

        self.test_obj.parse_config()

        self.test_obj.parsable_items.assert_called()
        self.test_obj.read_config_file.assert_called_with('c/config.txt',
                'items')
        self.test_obj.validate_config.assert_called_with('config')
        self.test_obj.populate_config.assert_called_with('config')


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(ConfigTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
