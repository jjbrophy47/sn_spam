"""
Tests the relational module.
"""
import os
import unittest
import pandas as pd
import mock
from .context import relational
from .context import config
from .context import psl
from .context import tuffy
from .context import test_utils as tu


class RelationalTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        mock_psl_obj = mock.Mock(psl.PSL)
        mock_tuffy_obj = mock.Mock(tuffy.Tuffy)
        self.test_obj = relational.Relational(config_obj, mock_psl_obj,
                mock_tuffy_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        # setup
        result = self.test_obj

        # assert
        self.assertTrue(isinstance(result.config_obj, config.Config))

    def test_define_file_folders(self):
        # setup
        os.path.exists = mock.Mock(return_value=False)
        os.makedirs = mock.Mock()

        # test
        result = self.test_obj.define_file_folders()

        # assert
        os.path.exists.assert_called_with('rel/psl/data/soundcloud/')
        os.makedirs.assert_called()
        self.assertTrue(result[0] == 'rel/psl/')
        self.assertTrue(result[1] == 'rel/psl/data/soundcloud/')
        self.assertTrue(result[2] == 'rel/tuffy/')
        self.assertTrue(result[3] == 'ind/data/soundcloud/folds/')
        self.assertTrue(result[4] == 'ind/output/soundcloud/predictions/')

    def test_check_dataframes_none(self):
        pd.read_csv = mock.Mock()
        pd.read_csv.side_effect = ['val_df', 'test_df']

        result = self.test_obj.check_dataframes(None, 'df', 'folds/')

        exp = [mock.call('folds/val_1.csv'), mock.call('folds/test_1.csv')]
        self.assertTrue(result[0] == 'val_df')
        self.assertTrue(result[1] == 'test_df')
        self.assertTrue(pd.read_csv.call_args_list == exp)

    def test_merge_ind_preds(self):
        test_df = tu.sample_df(10)
        test_df.columns = ['com_id', 'rando']
        pred_df = tu.sample_df(10)
        pd.read_csv = mock.Mock(return_value=pred_df)

        result = self.test_obj.merge_ind_preds(test_df, 'test', 'ind_pred/')

        self.assertTrue(len(list(result)) == 3)
        self.assertTrue(len(result) == 10)
        pd.read_csv.assert_called_with('ind_pred/test_1_preds.csv')

    def test_compile_reasoning_engine(self):
        folders = ('psl/', 'a/', 'b/', 'c/', 'd/')
        self.test_obj.define_file_folders = mock.Mock(return_value=folders)
        self.test_obj.psl_obj.compile = mock.Mock()

        self.test_obj.compile_reasoning_engine()

        self.test_obj.define_file_folders.assert_called()
        self.test_obj.psl_obj.compile.assert_called_with('psl/')

    def test_main(self):
        folders = ('a/', 'b/', 'c/', 'd/', 'e/')
        self.test_obj.define_file_folders = mock.Mock(return_value=folders)
        self.test_obj.check_dataframes = mock.Mock(return_value=('v', 't'))
        self.test_obj.merge_ind_preds = mock.Mock()
        self.test_obj.merge_ind_preds.side_effect = ['v_df', 't_df']
        self.test_obj.run_relational_model = mock.Mock()

        self.test_obj.main('v_df', 't_df')

        self.test_obj.define_file_folders.assert_called()
        self.test_obj.check_dataframes.assert_called_with('v_df', 't_df', 'd/')
        self.assertTrue(self.test_obj.merge_ind_preds.call_args_list ==
                [mock.call('v', 'val', 'e/'),
                mock.call('t', 'test', 'e/')])
        self.test_obj.run_relational_model.assert_called_with('v_df',
                't_df', 'a/', 'b/', 'c/')

    def test_run_psl(self):
        self.test_obj.psl_obj.clear_data = mock.Mock()
        self.test_obj.psl_obj.gen_predicates = mock.Mock()
        self.test_obj.psl_obj.gen_model = mock.Mock()
        self.test_obj.psl_obj.run = mock.Mock()

        self.test_obj.run_psl('v_df', 't_df', 'psl/', 'psl_data/')

        self.test_obj.psl_obj.clear_data.assert_called_with('psl_data/')
        self.assertTrue(self.test_obj.psl_obj.gen_predicates.call_args_list ==
                [mock.call('v_df', 'val', 'psl_data/'),
                mock.call('t_df', 'test', 'psl_data/')])
        self.test_obj.psl_obj.gen_model.assert_called_with('psl_data/')
        self.test_obj.psl_obj.run.assert_called_with('psl/')

    def test_run_tuffy(self):
        self.test_obj.tuffy_obj.clear_data = mock.Mock()
        self.test_obj.tuffy_obj.gen_predicates = mock.Mock()
        self.test_obj.tuffy_obj.run = mock.Mock()
        self.test_obj.tuffy_obj.parse_output = mock.Mock(return_value='p_df')
        self.test_obj.tuffy_obj.evaluate = mock.Mock()

        self.test_obj.run_tuffy('v_df', 't_df', 't/')

        self.test_obj.tuffy_obj.clear_data.assert_called_with('t/')
        self.assertTrue(self.test_obj.tuffy_obj.gen_predicates.call_args_list
                == [mock.call('v_df', 'val', 't/'),
                mock.call('t_df', 'test', 't/')])
        self.test_obj.tuffy_obj.run.assert_called_with('t/')
        self.test_obj.tuffy_obj.parse_output.assert_called_with('t/')
        self.test_obj.tuffy_obj.evaluate.assert_called_with('t_df', 'p_df')

    def test_run_relational_model_psl(self):
        self.test_obj.run_psl = mock.Mock()
        self.test_obj.run_tuffy = mock.Mock()

        self.test_obj.run_relational_model('v_df', 't_df', 'p/', 'pd/', 't/')

        self.test_obj.run_psl.assert_called_with('v_df', 't_df', 'p/', 'pd/')
        self.test_obj.run_tuffy.assert_not_called()

    def test_run_relational_model_tuffy(self):
        self.test_obj.config_obj.engine = 'tuffy'
        self.test_obj.run_psl = mock.Mock()
        self.test_obj.run_tuffy = mock.Mock()

        self.test_obj.run_relational_model('v_df', 't_df', 'p/', 'pd/', 't/')

        self.test_obj.run_tuffy.assert_called_with('v_df', 't_df', 't/')
        self.test_obj.run_psl.assert_not_called()


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(RelationalTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
