"""
Module to test one test set with increasing training data.
"""


class Training_Experiment:
    """Handles all operations to test the learning curve of the models."""

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
    def divide_data_into_subsets(self, growth_factor=2,
            train_start_size=100, train_split=0.7, val_split=0.3):
        """Splits the data into partitions, keeping the same test set
                but increasing the training data.
        growth_factor: number to multiply to increase the training size.
        train_start_size: amount of data to start training with.
        train_split: amount of training data to train the independent model.
        val_split: amount of training data to train the relational model.
        Returns a list of tuples containing the new start point, training
                size, validation size, and the experiment number."""
        if train_split + val_split != 1.0:
            print('train and val splits must add up to 1.0, exiting...')
            exit(0)

        test_size = self.config_obj.end - self.config_obj.start
        data_size = test_size + train_start_size

        start = self.config_obj.start - train_start_size
        fold = int(self.config_obj.fold)
        subsets = []

        while True:
            if start < 0:
                data_size = self.config_obj.end - 0
                train_data = data_size - test_size
                train_param = (train_data * train_split) / data_size
                val_param = (train_data * val_split) / data_size
                subset = (0, train_param, val_param, str(fold))
                subsets.append(subset)
                break

            train_data = data_size - test_size
            train_param = (train_data * train_split) / data_size
            val_param = (train_data * val_split) / data_size
            subset = (start, train_param, val_param, str(fold))
            subsets.append(subset)

            start -= (train_data * growth_factor) - train_data
            data_size = self.config_obj.end - start
            fold += 1
        return subsets

    def run_experiment(self, subsets):
        """Configures the application based on the data subsets, and then runs
                the independent and relational models."""
        for start, train_size, val_size, fold in subsets:
            self.change_config_parameters(start, train_size, val_size, fold)
            self.single_run()

    # private
    def single_run(self):
        """Operations to run the independent model, train the relational model
                from those predictions, and then do joint prediction using
                the relational model on the test set."""
        val_df, test_df = self.runner_obj.run_independent()
        self.runner_obj.run_purity(test_df)
        self.change_config_rel_op(train=True)
        self.runner_obj.run_relational(val_df, test_df)
        self.change_config_rel_op(train=False)
        self.runner_obj.run_relational(val_df, test_df)
        self.runner_obj.run_evaluation(test_df)

    def change_config_parameters(self, start, train_size, val_size, fold):
        """Changes configuration options for the start of the data subset,
                training size, validation size, and experiment number."""
        self.config_obj.start = start
        self.config_obj.train_size = train_size
        self.config_obj.val_size = val_size
        self.config_obj.fold = fold

    def change_config_rel_op(self, train=True):
        """Changes to training or inference for the relational model."""
        self.config_obj.infer = not train
