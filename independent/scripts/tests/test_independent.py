"""
Tests the independent module.
"""
import os
import unittest
import pandas as pd
import mock
from .context import independent
from .context import config
from .context import classification
from .context import util
from .context import test_utils as tu


class IndependentTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        mock_classification_obj = mock.Mock(classification.Classification)
        mock_util_obj = mock.Mock(util.Util)
        self.test_obj = independent.Independent(config_obj,
                mock_classification_obj, mock_util_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        # setup
        test_obj = self.test_obj

        # assert
        self.assertTrue(isinstance(test_obj.config_obj, config.Config))
        self.assertTrue(isinstance(test_obj.classification_obj,
                classification.Classification))
        self.assertTrue(isinstance(test_obj.util_obj, util.Util))

    def test_define_file_folders(self):
        os.path.exists = mock.Mock(return_value=False)
        os.makedirs = mock.Mock()

        # test
        result = self.test_obj.define_file_folders()

        # assert
        self.assertTrue(result[0] == 'ind/data/soundcloud/')
        self.assertTrue(result[1] == 'ind/data/soundcloud/folds/')
        self.assertTrue(os.path.exists.called)
        self.assertTrue(os.makedirs.called)

    def test_read_file(self):
        df = tu.simple_df()
        pd.read_csv = mock.Mock(return_value=df)
        self.test_obj.config_obj.end = 7

        result = self.test_obj.read_file('boogers.txt')

        pd.read_csv.assert_called_with('boogers.txt', lineterminator='\n',
                nrows=7)
        self.assertTrue(result.equals(tu.simple_df()))

    def test_load_convenience_files_doesnt_exist(self):
        df = tu.simple_df()
        df.columns = ['com_id']
        os.path.exists = mock.Mock(return_value=False)
        pd.read_csv = mock.Mock()

        result = self.test_obj.load_convenience_files(df, 'relation.tsv')

        os.path.exists.assert_called_with('relation.tsv')
        pd.read_csv.assert_not_called()
        self.assertTrue(list(result) == ['com_id'])
        self.assertTrue(len(result) == 10)

    def test_load_convenience_files_exists(self):
        df = tu.simple_df()
        df.columns = ['com_id']
        df2 = tu.simple_df()
        df2.columns = ['com_id']
        df2['random'] = 69
        os.path.exists = mock.Mock(return_value=True)
        pd.read_csv = mock.Mock(return_value=df2)

        result = self.test_obj.load_convenience_files(df, 'relation.tsv')

        os.path.exists.assert_called_with('relation.tsv')
        pd.read_csv.assert_called_with('relation.tsv', lineterminator='\n',
                sep='\t')
        self.assertTrue(list(result) == ['com_id', 'random'])
        self.assertTrue(len(result) == 10)

    def test_split_coms(self):
        df = tu.simple_df()
        self.test_obj.config_obj.start = 2
        self.test_obj.config_obj.train_size = 0.5

        result = self.test_obj.split_coms(df)

        self.assertTrue(len(result[0]) == 4)
        self.assertTrue(len(result[1]) == 2)
        self.assertTrue(len(result[2]) == 2)

    def test_write_folds(self):
        val_df = tu.simple_df()
        test_df = tu.simple_df()
        val_df.to_csv = mock.Mock()
        test_df.to_csv = mock.Mock()

        self.test_obj.write_folds(val_df, test_df, 'data/')

        val_df.to_csv.assert_called_with('data/val_1.csv', index=None,
                line_terminator='\n')
        test_df.to_csv.assert_called_with('data/test_1.csv', index=None,
                line_terminator='\n')

    def test_print_subsets(self):
        train_df = tu.simple_df()
        val_df = tu.simple_df()
        test_df = tu.simple_df()
        train_df.columns = ['label']
        val_df.columns = ['label']
        test_df.columns = ['label']
        self.test_obj.util_obj.div0 = mock.Mock(return_value=0.69)

        self.test_obj.print_subsets(train_df, val_df, test_df)

        self.test_obj.util_obj.div0.assert_called_with(1, 10)
        self.assertTrue(self.test_obj.util_obj.div0.call_count == 3)

    def test_main(self):
        self.test_obj.define_file_folders = mock.Mock(return_value=(
                'a/', 'b/'))
        self.test_obj.util_obj.get_comments_filename = mock.Mock(
                return_value='fname')
        self.test_obj.read_file = mock.Mock(return_value='df')
        self.test_obj.load_convenience_files = mock.Mock(return_value='df2')
        self.test_obj.split_coms = mock.Mock(return_value=('tr', 'va', 'te'))
        self.test_obj.write_folds = mock.Mock()
        self.test_obj.print_subsets = mock.Mock()
        self.test_obj.classification_obj.main = mock.Mock()

        result = self.test_obj.main()

        self.test_obj.define_file_folders.assert_called()
        self.test_obj.util_obj.get_comments_filename.assert_called_with(False)
        self.test_obj.read_file.assert_called_with('a/fname')
        self.test_obj.load_convenience_files.assert_called_with('df',
                'a/intext.tsv')
        self.test_obj.split_coms.assert_called_with('df2')
        self.test_obj.write_folds.assert_called_with('va', 'te', 'b/')
        self.test_obj.print_subsets.assert_called_with('tr', 'va', 'te')
        self.test_obj.classification_obj.main.assert_called_with('tr', 'va',
                'te')
        self.assertTrue(result[0] == 'va')
        self.assertTrue(result[1] == 'te')


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(IndependentTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
