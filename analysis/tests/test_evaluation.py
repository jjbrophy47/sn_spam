"""
Tests the evaluation module.
"""
import os
import unittest
import pandas as pd
import mock
from .context import evaluation
from .context import config
from .context import generator
from .context import util
from .context import test_utils as tu


class EvaluationTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        generator_obj = generator.Generator()
        mock_util_obj = mock.Mock(util.Util)
        self.test_obj = evaluation.Evaluation(config_obj, generator_obj,
                mock_util_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        # setup
        result = self.test_obj

        # assert
        self.assertTrue(isinstance(result.config_obj, config.Config))
        self.assertTrue(isinstance(result.util_obj, util.Util))

    def test_settings(self):
        self.test_obj.util_obj.set_noise_limit = mock.Mock()

        self.test_obj.settings()

        self.test_obj.util_obj.set_noise_limit.assert_called_with(0.0025)

    def test_define_file_folders(self):
        os.makedirs = mock.Mock()

        result = self.test_obj.define_file_folders()

        self.assertTrue(result[0] == 'ind/data/soundcloud/')
        self.assertTrue(result[1] == 'ind/output/soundcloud/predictions/')
        self.assertTrue(result[2] == 'rel/output/soundcloud/predictions/')
        self.assertTrue(result[3] == 'rel/output/soundcloud/images/')

    def test_read_predictions(self):
        df = tu.sample_df(10)
        pd.read_csv = mock.Mock(return_value=df)

        result = self.test_obj.read_predictions('ind/', 'rel/')

        expected = [mock.call('ind/test_1_preds.csv'),
                mock.call('rel/predictions_1.csv')]
        self.assertTrue(pd.read_csv.call_args_list == expected)
        self.assertTrue(result[0].equals(tu.sample_df(10)))
        self.assertTrue(result[1].equals(tu.sample_df(10)))

    def test_merge_predictions(self):
        df = tu.sample_df(10)
        df1 = tu.sample_df(10)
        df1.columns = ['com_id', 'ip']
        df2 = tu.sample_df(10)
        df2.columns = ['com_id', 'rp']

        result = self.test_obj.merge_predictions(df, df1, df2)

        self.assertTrue(list(result) == ['com_id', 'random', 'ip', 'rp'])
        self.assertTrue(len(result) == 10)

    def test_apply_noise(self):
        df = tu.sample_df(10)
        df.columns = ['com_id', 'ind_pred']
        df['rel_pred'] = 100
        df2 = df.copy()

        result = self.test_obj.apply_noise(df)

        self.assertTrue(len(result) == 10)
        self.assertTrue(not result.equals(df2))

    def test_gen_group_ids(self):
        df = tu.sample_df(4)
        filled_df = tu.sample_df(4)
        df.copy = mock.Mock(return_value=filled_df)
        self.test_obj.generator_obj.gen_group_id = mock.Mock()
        self.test_obj.generator_obj.gen_group_id.side_effect = ['r1_df',
                'r2_df']

        result = self.test_obj.gen_group_ids(df)

        self.assertTrue(self.test_obj.generator_obj.gen_group_id.
                call_args_list == [mock.call(df, 'text_id'),
                mock.call('r1_df', 'user_id')])
        self.assertTrue(result == 'r2_df')

    def test_relational_comments_only(self):
        exp_df = tu.sample_relational_df()

        result = self.test_obj.relational_comments_only(exp_df)

        # exp_df loses indexes 3 & 4, com_ids 4 & 5.
        exp = pd.Series([1, 2, 3, 6, 7, 8], index=[0, 1, 2, 5, 6, 7])
        self.assertTrue(len(result) == 6)
        self.assertTrue(result['com_id'].equals(exp))

    def test_compute_scores(self):
        df = tu.sample_df(10)
        df['pred'] = [0.1, 0.7, 0.3, 0.4, 0.7, 0.8, 0.9, 0.2, 0.77, 0.88]
        df['label'] = [0, 1, 0, 1, 1, 1, 1, 0, 1, 1]

        result = self.test_obj.compute_scores(df, 'pred')

        self.assertTrue(result[0] == 0.99999999999999978)  # aupr
        self.assertTrue(result[1] == 1.0)  # auroc
        self.assertTrue(len(result[2]) == 7)  # recalls
        self.assertTrue(len(result[3]) == 7)  # precisions
        self.assertTrue(result[4] == 1.0)  # n-aupr

    def test_evaluate(self):
        df = tu.sample_df(10)
        df.copy = mock.Mock(return_value='t_df')
        self.test_obj.settings = mock.Mock()
        self.test_obj.define_file_folders = mock.Mock(return_value=('a/',
                'b/', 'c/', 'd/'))
        self.test_obj.read_predictions = mock.Mock(return_value=('ind_df',
                'rel_df'))
        self.test_obj.merge_predictions = mock.Mock(return_value='m_df')
        self.test_obj.apply_noise = mock.Mock(return_value='noise_df')
        self.test_obj.compute_scores = mock.Mock()
        self.test_obj.compute_scores.side_effect = [('i1', 'i2', 'i3', 'i4',
                'i5'), ('r1', 'r2', 'r3', 'r4', 'r5')]
        self.test_obj.print_scores = mock.Mock()
        self.test_obj.util_obj.plot_pr_curve = mock.Mock()

        self.test_obj.evaluate(df)

        exp1 = [mock.call('noise_df', 'ind_pred'),
                mock.call('noise_df', 'rel_pred')]
        exp2 = [mock.call('Independent', 'i1', 'i2', 'i5'),
                mock.call('Relational', 'r1', 'r2', 'r5')]
        exp3 = [mock.call('Independent', '', 'i3', 'i4', 'i1'),
                mock.call('Relational', 'd/pr_1', 'r3', 'r4', 'r1',
                line='--', save=True)]
        df.copy.assert_called()
        self.test_obj.settings.assert_called()
        self.test_obj.define_file_folders.assert_called()
        self.test_obj.read_predictions.assert_called_with('b/', 'c/')
        self.test_obj.merge_predictions.assert_called_with('t_df', 'ind_df',
                'rel_df')
        self.test_obj.apply_noise.assert_called_with('m_df')
        self.assertTrue(self.test_obj.compute_scores.call_args_list == exp1)
        self.assertTrue(self.test_obj.print_scores.call_args_list == exp2)
        self.assertTrue(self.test_obj.util_obj.plot_pr_curve.call_args_list ==
                exp3)

    def test_evaluate_modified(self):
        df = tu.sample_df(10)
        df.copy = mock.Mock(return_value='t_df')
        self.test_obj.settings = mock.Mock()
        self.test_obj.define_file_folders = mock.Mock(return_value=('a/',
                'b/', 'c/', 'd/'))
        self.test_obj.read_predictions = mock.Mock(return_value=('ind_df',
                'rel_df'))
        self.test_obj.merge_predictions = mock.Mock(return_value='m_df')
        self.test_obj.read_modified = mock.Mock(return_value='mod_df')
        self.test_obj.filter = mock.Mock(return_value='filt_df')
        self.test_obj.apply_noise = mock.Mock(return_value='noise_df')
        self.test_obj.compute_scores = mock.Mock()
        self.test_obj.compute_scores.side_effect = [('i1', 'i2', 'i3', 'i4',
                'i5'), ('r1', 'r2', 'r3', 'r4', 'r5')]
        self.test_obj.print_scores = mock.Mock()
        self.test_obj.util_obj.plot_pr_curve = mock.Mock()

        self.test_obj.evaluate_modified(df)

        exp1 = [mock.call('noise_df', 'ind_pred'),
                mock.call('noise_df', 'rel_pred')]
        exp2 = [mock.call('Independent', 'i1', 'i2', 'i5'),
                mock.call('Relational', 'r1', 'r2', 'r5')]
        exp3 = [mock.call('Independent', '', 'i3', 'i4', 'i1'),
                mock.call('Relational', 'd/pr_1', 'r3', 'r4', 'r1',
                line='--', save=True)]
        df.copy.assert_called()
        self.test_obj.settings.assert_called()
        self.test_obj.define_file_folders.assert_called()
        self.test_obj.read_predictions.assert_called_with('b/', 'c/')
        self.test_obj.merge_predictions.assert_called_with('t_df', 'ind_df',
                'rel_df')
        self.test_obj.read_modified.assert_called_with('a/')
        self.test_obj.filter.assert_called_with('m_df', 'mod_df')
        self.test_obj.apply_noise.assert_called_with('filt_df')
        self.assertTrue(self.test_obj.compute_scores.call_args_list == exp1)
        self.assertTrue(self.test_obj.print_scores.call_args_list == exp2)
        self.assertTrue(self.test_obj.util_obj.plot_pr_curve.call_args_list ==
                exp3)


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(EvaluationTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
