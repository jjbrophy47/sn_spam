"""
Module to test different test sets in a domain.
"""
import os
import pandas as pd


class Subsets_Experiment:
    """Handles all operations to test different parts of the data."""

    def __init__(self, config_obj, runner_obj, modified=True, pseudo=True):
        """Initializes object dependencies."""

        self.config_obj = config_obj
        """User settings."""
        self.runner_obj = runner_obj
        """Runs different parts of the application."""
        self.config_obj.modified = modified
        """Boolean to use relabeled data if True."""
        self.config_obj.pseudo = pseudo
        """Boolean to use all features if True."""

    # public
    def divide_data_into_subsets(self, num_subsets=10):
        """Partitions the data into a specified number of subsets to run the
                independent and relational models on.
        num_subsets: number of partitions to split the data into.
        Returns a list of tuples containing the start and end of the data
                subset, as well as the experiment number."""
        data_size = self.config_obj.end - self.config_obj.start
        subset_size = data_size / num_subsets
        subsets = []

        for i in range(num_subsets):
            start = int(self.config_obj.start + (i * subset_size))
            end = int(start + subset_size)
            fold = str(int(self.config_obj.fold) + i)
            subset = (start, end, fold)
            subsets.append(subset)
        return subsets

    def run_experiment(self, subsets):
        """Configures the application based on the data subsets, and then runs
                the independent and relational models."""
        self._clear_data()

        rel_dir = self.config_obj.rel_dir
        domain = self.config_obj.domain
        out_dir = rel_dir + 'output/' + domain + '/subsets_exp/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        rows = []
        for start, end, fold in subsets:
            self.change_config_parameters(start, end, fold)
            score_dict = self.single_run()

            row = tuple()
            for model_name, scores in score_dict.items():
                row += scores
            rows.append(row)

            columns = []
            for key, value in score_dict.items():
                for i in range(len(value)):
                    column = key + '_' + str(i)
                    columns.append(column)

            df = pd.DataFrame(rows, columns=columns)
            df.to_csv(out_dir + 'results.csv', index=None)

    # private
    def _clear_data(self):
        ind_dir = self.config_obj.ind_dir
        rel_dir = self.config_obj.rel_dir
        domain = self.config_obj.domain

        fold_dir = ind_dir + '/data/' + domain + '/folds/'
        ind_pred_dir = ind_dir + '/output/' + domain + '/predictions/'
        rel_pred_dir = rel_dir + '/output/' + domain + '/predictions/'

        os.system('rm %s*.csv' % (fold_dir))
        os.system('rm %s*.csv' % (ind_pred_dir))
        os.system('rm %s*.csv' % (rel_pred_dir))

    def single_run(self):
        """Operations to run the independent model, train the relational model
                from those predictions, and then do joint prediction using
                the relational model on the test set."""

        # run independent
        val_df, test_df = self.runner_obj.run_independent()

        # run PSL training and inference
        self.change_config_engine(engine='psl')
        self.change_config_rel_op(train=True)
        self.runner_obj.run_relational(val_df, test_df)
        self.change_config_rel_op(train=False)
        self.runner_obj.run_relational(val_df, test_df)

        # run MRF loopy bp inference
        self.change_config_engine(engine='mrf')
        self.runner_obj.run_relational(val_df, test_df)

        # evaluate predictions from each method
        score_dict = self.runner_obj.run_evaluation(test_df)
        return score_dict

    def change_config_parameters(self, start, end, fold):
        """Changes the start, end and experiment number in the configuration
                options."""
        self.config_obj.start = start
        self.config_obj.end = end
        self.config_obj.fold = fold

    def change_config_engine(self, engine='psl'):
        self.config_obj.engine = engine

    def change_config_rel_op(self, train=True):
        """Changes to training or inference for the relational model."""
        self.config_obj.infer = not train
