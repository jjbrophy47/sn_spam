"""
Tests the classification module.
"""
import os
import unittest
import numpy as np
import scipy
import mock
from scipy.sparse import csr_matrix
from .context import classification
from .context import config
from .context import content_features
from .context import graph_features
from .context import relational_features
from .context import util
from .context import test_utils as tu


class ClassificationTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        mock_content_features_obj = mock.Mock(content_features.ContentFeatures)
        mock_graph_features_obj = mock.Mock(graph_features.GraphFeatures)
        mock_rel_feats_obj = mock.Mock(relational_features.RelationalFeatures)
        mock_util_obj = mock.Mock(util.Util)
        self.test_obj = classification.Classification(config_obj,
                mock_content_features_obj, mock_graph_features_obj,
                mock_rel_feats_obj, mock_util_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        # setup
        test_obj = self.test_obj

        # assert
        self.assertTrue(isinstance(test_obj.config_obj, config.Config))
        self.assertTrue(isinstance(test_obj.cf_obj,
                                   content_features.ContentFeatures))
        self.assertTrue(isinstance(test_obj.gf_obj,
                                   graph_features.GraphFeatures))
        self.assertTrue(isinstance(test_obj.rf_obj,
                                   relational_features.RelationalFeatures))
        self.assertTrue(isinstance(test_obj.util_obj, util.Util))

    def test_define_file_folders(self):
        os.path.exists = mock.Mock(return_value=False)
        os.makedirs = mock.Mock()

        # test
        result = self.test_obj.define_file_folders()

        # assert
        self.assertTrue(result[0] == 'ind/output/soundcloud/images/')
        self.assertTrue(result[1] == 'ind/output/soundcloud/predictions/')
        self.assertTrue(result[2] == 'ind/output/soundcloud/models/')
        self.assertTrue(os.path.exists.call_count == 3)
        self.assertTrue(os.makedirs.call_count == 3)

    def test_merge_no_graph_features(self):
        coms_df = tu.sample_df_with_com_id(10)
        c_df = tu.sample_df_with_com_id(10)
        r_df = tu.sample_df_with_com_id(10)
        g_df = None

        result = self.test_obj.merge(coms_df, c_df, g_df, r_df)

        self.assertTrue(len(list(result)) == 4)
        self.assertTrue(len(result) == 10)

    def test_merge_with_graph_features(self):
        coms_df = tu.sample_df_with_com_id_and_user_id(10)
        c_df = tu.sample_df_with_com_id(10)
        r_df = tu.sample_df_with_com_id(10)
        g_df = tu.sample_df_with_user_id(5)

        result = self.test_obj.merge(coms_df, c_df, g_df, r_df)

        self.assertTrue(len(list(result)) == 5)
        self.assertTrue(len(result) == 10)
        self.assertFalse(result.isnull().values.any())

    def test_drop_columns(self):
        coms_df = tu.sample_df_with_com_id_and_user_id(10)
        c_df = tu.sample_df_with_com_id(10)
        r_df = tu.sample_df_with_com_id(10)
        g_df = tu.sample_df_with_user_id(5)
        feats_df = self.test_obj.merge(coms_df, c_df, g_df, r_df)
        features = ['random_x', 'random_y']

        result = self.test_obj.drop_columns(feats_df, features)

        self.assertTrue(list(result) == ['random_x', 'random_y'])

    def test_dataframe_to_matrix(self):
        coms_df = tu.sample_df_with_com_id_and_user_id(10)
        c_df = tu.sample_df_with_com_id(10)
        r_df = tu.sample_df_with_com_id(10)
        g_df = tu.sample_df_with_user_id(5)
        feats_df = self.test_obj.merge(coms_df, c_df, g_df, r_df)

        result = self.test_obj.dataframe_to_matrix(feats_df)

        self.assertTrue(type(result) == scipy.sparse.csr.csr_matrix)
        self.assertTrue(result.shape == (10, 5))

    def test_stack_matrices(self):
        feats_m = tu.sample_df_with_com_id_and_user_id(10).as_matrix()
        c_csr = csr_matrix(tu.sample_df_with_com_id(10).as_matrix())

        result = self.test_obj.stack_matrices(feats_m, c_csr)

        self.assertTrue(type(result) == scipy.sparse.csr.csr_matrix)
        self.assertTrue(result.shape == (10, 4))

    def test_extract_labels(self):
        coms_df = tu.sample_df_with_com_id_and_user_id(10)
        coms_df.columns = ['com_id', 'label']

        result = self.test_obj.extract_labels(coms_df)

        self.assertTrue(type(result[0]) == np.ndarray)
        self.assertTrue(result[0].shape == (10,))
        self.assertTrue(len(result[0]) == 10)

    def test_transform(self):
        self.test_obj.merge = mock.Mock(return_value='f_df')
        self.test_obj.drop_columns = mock.Mock(return_value='f_df2')
        self.test_obj.dataframe_to_matrix = mock.Mock(return_value='f_m')
        self.test_obj.stack_matrices = mock.Mock(return_value='x')
        self.test_obj.extract_labels = mock.Mock(return_value=('y', 'ids'))

        result = self.test_obj.transform('df', 'c_m', 'c_df', 'g_df', 'r_df',
                'feats_list')

        self.test_obj.merge.assert_called_with('df', 'c_df', 'g_df', 'r_df')
        self.test_obj.drop_columns.assert_called_with('f_df', 'feats_list')
        self.test_obj.dataframe_to_matrix.assert_called_with('f_df2')
        self.test_obj.stack_matrices.assert_called_with('f_m', 'c_m')
        self.test_obj.extract_labels.assert_called_with('df')
        self.assertTrue(result == ('x', 'y', 'ids'))

    def test_main(self):
        tr_df = mock.Mock()
        val_df = mock.Mock()
        test_df = mock.Mock()
        self.test_obj.define_file_folders = mock.Mock(return_value=('b/',
                'c/', 'd/'))
        self.test_obj.cf_obj.build = mock.Mock(return_value=('c_tr_m',
                'c_va_m', 'c_te_m', 'c_df', 'c_f'))
        self.test_obj.gf_obj.build = mock.Mock(return_value=('g_df', 'g_f'))
        self.test_obj.rf_obj.build = mock.Mock(return_value=('r_df', 'r_f'))
        self.test_obj.util_obj.start = mock.Mock()
        self.test_obj.transform = mock.Mock(return_value=('x', 'y', 'z'))
        self.test_obj.util_obj.end = mock.Mock()
        self.test_obj.util_obj.classify = mock.Mock()

        self.test_obj.main(tr_df, val_df, test_df)

        expected = [mock.call(tr_df, 'c_tr_m', 'c_df', 'g_df', 'r_df',
                'c_fg_fr_f'), mock.call(val_df, 'c_va_m', 'c_df', 'g_df',
                'r_df', 'c_fg_fr_f'), mock.call(test_df, 'c_te_m', 'c_df',
                'g_df', 'r_df', 'c_fg_fr_f')]
        self.test_obj.define_file_folders.assert_called()
        self.test_obj.cf_obj.build.assert_called_with(tr_df, val_df, test_df)
        self.test_obj.gf_obj.build.assert_called_with(tr_df, val_df, test_df)
        self.test_obj.rf_obj.build.assert_called_with(tr_df, val_df, test_df)
        self.test_obj.util_obj.start.assert_called_with('merging features...')
        self.assertTrue(self.test_obj.transform.call_args_list == expected)
        self.test_obj.util_obj.end.assert_called()
        self.test_obj.util_obj.classify.assert_called_with('x', 'y',
                'x', 'y', 'x', 'y', 'z', 'z', '1', 'c_fg_fr_f', 'all',
                'b/', 'c/', 'd/', classifier='lr', save_feat_plot=True)


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(ClassificationTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
