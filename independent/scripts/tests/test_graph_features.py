"""
Tests the graph_features module.
"""
import mock
import unittest
import pandas as pd
from .context import graph_features
from .context import config
from .context import util
from .context import test_utils as tu


class GraphFeaturesTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        util_obj = util.Util()
        self.test_obj = graph_features.GraphFeatures(config_obj, util_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        result = self.test_obj

        self.assertTrue(isinstance(result.config_obj, config.Config))

    def test_build_wrong_domain(self):
        self.test_obj.config_obj.domain = 'youtube'

        result = self.test_obj.build('tr', 'te')

        self.assertTrue(result == (None, []))

    def test_build_correct_domain_with_graph_fetures(self):
        feats_df = tu.sample_df(10)
        feats_df.columns = ['user_id', 'random']
        self.test_obj.util_obj.start = mock.Mock()
        self.test_obj.util_obj.end = mock.Mock()
        self.test_obj.define_file_folders = mock.Mock(return_value=('data/',
                'gl/', 'feat/'))
        self.test_obj.util_obj.check_file = mock.Mock(return_value=True)
        pd.read_csv = mock.Mock(return_value=feats_df)

        result = self.test_obj.build('tr_df', 'te_df')

        exp_start = 'loading graph features...'
        exp_file = 'feat/graph_features.csv'
        self.test_obj.define_file_folders.assert_called()
        self.test_obj.util_obj.check_file.assert_called_with(exp_file)
        self.test_obj.util_obj.start.assert_called_with(exp_start)
        pd.read_csv.assert_called_with('feat/graph_features.csv')
        self.test_obj.util_obj.end.assert_called()
        self.assertTrue(list(result[0]) == ['user_id', 'random'])
        self.assertTrue(len(result[0]) == 10)
        self.assertTrue(result[1] == ['random'])

    @mock.patch('os.makedirs')
    def test_define_file_folders(self, mock_makedirs):
        result = self.test_obj.define_file_folders()

        gl_f = 'ind/data/soundcloud/graphlab/'
        feat_f = 'ind/output/soundcloud/features/'
        self.assertTrue(result[0] == 'ind/data/soundcloud/')
        self.assertTrue(result[1] == gl_f)
        self.assertTrue(result[2] == feat_f)
        self.assertTrue(mock_makedirs.call_args_list == [mock.call(feat_f),
                mock.call(gl_f)])


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(
            GraphFeaturesTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
