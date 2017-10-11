"""
Module to test one test set with increasing training data.
"""
import pandas as pd


class Training_Experiment:
    """Handles all operations to test the learning curve of the models."""

    def __init__(self, config_obj, runner_obj, modified=True, pseudo=True,
            super_train=False):
        """Initializes object dependencies."""

        self.config_obj = config_obj
        """User settings."""
        self.runner_obj = runner_obj
        """Runs different parts of the application."""
        self.config_obj.modified = modified
        """Boolean to use relabeled data if True."""
        self.config_obj.pseudo = pseudo
        """Boolean to use all features if True."""
        self.super_train = super_train
        """Boolean to use both train and val for training if True."""

    # public
    def divide_data_into_subsets(self, growth_factor=2, val_size=100):
        """Splits the data into partitions, keeping the same test set
                but increasing the training data.
        growth_factor: number to multiply to increase the training size.
        val_size: amount of training data to train the relational model.
        Returns a list of folds to run training and inference on."""
        domain = self.config_obj.domain
        fold = self.config_obj.fold
        ind_dir = self.config_obj.ind_dir
        start = self.config_obj.start
        end = self.config_obj.end
        val_param = self.config_obj.val_size
        avail_training_data = int((end - start) * val_param)
        folds = []

        # run the independent model to get predictions for val and test sets.
        val, test = self.independent_run()

        # merge independent predictions
        pred_f = ind_dir + 'output/' + domain + '/predictions/'
        val_pred = pd.read_csv(pred_f + 'val_' + fold + '_preds.csv')
        test_pred = pd.read_csv(pred_f + 'test_' + fold + '_preds.csv')
        val_df = val.merge(val_pred, on='com_id', how='left')
        test_df = test.merge(test_pred, on='com_id', how='left')

        while val_size < avail_training_data:
            self.create_fold(val_df, test_df, val_size, fold)
            folds.append(fold)
            val_size *= growth_factor
            fold = str(int(fold) + 1)
        return folds

    def run_experiment(self, folds):
        """Train and infer on each fold of data.
        folds: list of experiment numbers."""
        for fold in folds:
            self.change_config_fold(fold)
            val_df, test_df = self.read_fold(fold)
            self.relational_run(val_df, test_df)

    # private
    def independent_run(self):
        """Run just the independent model"""
        val_df, test_df = self.runner_obj.run_independent()
        return val_df, test_df

    def create_fold(self, val_df, test_df, val_size, fold):
        """Create a subset for an experiment fold and write the data to files.
        val_df: validation data.
        test_df: test data.
        val_size: amount of data to save for the experiment.
        fold: experiment number."""
        domain = self.config_obj.domain
        ind_dir = self.config_obj.ind_dir

        val_df = val_df.tail(val_size)
        val_fold = val_df.drop(['ind_pred'], axis=1)
        test_fold = test_df.drop(['ind_pred'], axis=1)

        fold_f = ind_dir + 'data/' + domain + '/folds/'
        val_fold.to_csv(fold_f + 'val_' + fold + '.csv', index=None,
                line_terminator='\n')
        test_fold.to_csv(fold_f + 'test_' + fold + '.csv', index=None,
                line_terminator='\n')

        pred_f = ind_dir + 'output/' + domain + '/predictions/'
        val_df.to_csv(pred_f + 'val_' + fold + '_preds.csv', index=None,
                line_terminator='\n', columns=['com_id', 'ind_pred'])
        test_df.to_csv(pred_f + 'test_' + fold + '_preds.csv', index=None,
                line_terminator='\n', columns=['com_id', 'ind_pred'])

        return val_fold, test_fold

    def read_fold(self, fold):
        """Reads the data for a specific fold.
        fold: experiment number."""
        domain = self.config_obj.domain
        ind_dir = self.config_obj.ind_dir
        fold_f = ind_dir + 'data/' + domain + '/folds/'

        val_df = pd.read_csv(fold_f + 'val_' + fold + '.csv')
        test_df = pd.read_csv(fold_f + 'test_' + fold + '.csv')
        return val_df, test_df

    def relational_run(self, val_df, test_df):
        """Operations to train the relational model and do joint prediction."""
        self.change_config_rel_op(train=True)
        self.runner_obj.run_relational(val_df, test_df)
        self.change_config_rel_op(train=False)
        self.runner_obj.run_relational(val_df, test_df)
        self.runner_obj.run_evaluation(test_df)

    def change_config_fold(self, fold):
        """Changes the experiment number.
        fold: experiment number."""
        self.config_obj.fold = fold

    def change_config_rel_op(self, train=True):
        """Changes to training or inference for the relational model."""
        self.config_obj.infer = not train
