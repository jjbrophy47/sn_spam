"""
Tests the training_exp module.
"""
import mock
import unittest
import numpy as np
from .context import training_exp
from .context import config
from .context import runner
from .context import test_utils as tu


class Training_ExperimentTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        mock_runner_obj = mock.Mock(runner.Runner)
        self.test_obj = training_exp.Training_Experiment(config_obj,
                mock_runner_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        # setup
        test_obj = self.test_obj

        # assert
        self.assertTrue(isinstance(test_obj.config_obj, config.Config))
        self.assertTrue(isinstance(test_obj.runner_obj, runner.Runner))
        self.assertTrue(test_obj.config_obj.modified)
        self.assertTrue(test_obj.config_obj.pseudo)

    def test_divide_data_into_subsets(self):
        self.test_obj.config_obj.end = 4000
        self.test_obj.config_obj.start = 3500
        self.test_obj.config_obj.fold = '0'

        result = self.test_obj.divide_data_into_subsets(growth_factor=2,
                val_size=100)

        exp_start = [0, 0, 0, 0]
        exp_train = [0.85, 0.825, 0.775, 0.675]
        exp_val = [0.025, 0.05, 0.1, 0.2]
        exp_fold = ['0', '1', '2', '3']
        res_start = [res[0] for res in result]
        res_train = [res[1] for res in result]
        res_val = [res[2] for res in result]
        res_fold = [res[3] for res in result]
        self.assertTrue(len(result) == 4)
        self.assertTrue(res_start == exp_start)
        self.assertTrue(np.allclose(res_train, exp_train, rtol=1e-4,
                atol=1e-1))
        self.assertTrue(np.allclose(res_val, exp_val, rtol=1e-4,
                atol=1e-1))
        self.assertTrue(res_fold == exp_fold)

    def test_run_experiment(self):
        subsets = [(1, 0.2, 0.1, '4'), (7, 0.3, 0.2, '88'),
                (7, 0.4, 0.3, '169')]
        self.test_obj.single_run = mock.Mock()
        self.test_obj.change_config_parameters = mock.Mock()

        self.test_obj.run_experiment(subsets)

        exp_ccp = [mock.call(1, 0.2, 0.1, '4'), mock.call(7, 0.3, 0.2, '88'),
                mock.call(7, 0.4, 0.3, '169')]
        self.assertTrue(self.test_obj.single_run.call_count == 3)
        self.assertTrue(self.test_obj.change_config_parameters.call_args_list
                == exp_ccp)

    def test_single_run(self):
        self.test_obj.runner_obj.run_independent = mock.Mock()
        self.test_obj.runner_obj.run_independent.return_value = ('v', 't')
        self.test_obj.runner_obj.run_purity = mock.Mock()
        self.test_obj.change_config_rel_op = mock.Mock()
        self.test_obj.runner_obj.run_relational = mock.Mock()
        self.test_obj.runner_obj.run_evaluation = mock.Mock()

        self.test_obj.single_run()

        exp_ccro = [mock.call(train=True), mock.call(train=False)]
        exp_rel = [mock.call('v', 't'), mock.call('v', 't')]
        self.test_obj.runner_obj.run_independent.assert_called_with()
        self.test_obj.runner_obj.run_purity.assert_called_with('t')
        self.assertTrue(self.test_obj.change_config_rel_op.call_args_list ==
                exp_ccro)
        self.assertTrue(self.test_obj.runner_obj.run_relational.call_args_list
                == exp_rel)
        self.test_obj.runner_obj.run_evaluation.assert_called_with('t')

    def test_change_config_parameters(self):
        self.test_obj.change_config_parameters(2, 0.22, 0.2, '69')

        self.assertTrue(self.test_obj.config_obj.start == 2)
        self.assertTrue(self.test_obj.config_obj.train_size == 0.22)
        self.assertTrue(self.test_obj.config_obj.val_size == 0.2)
        self.assertTrue(self.test_obj.config_obj.fold == '69')

    def test_change_config_rel_op(self):
        self.test_obj.change_config_rel_op(train=False)

        self.assertTrue(self.test_obj.config_obj.infer)


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(
            Training_ExperimentTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
