"""
Module to test different test sets in a domain.
"""


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
        for start, end, fold in subsets:
            self.change_config_parameters(start, end, fold)
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

    def change_config_parameters(self, start, end, fold):
        """Changes the start, end and experiment number in the configuration
                options."""
        self.config_obj.start = start
        self.config_obj.end = end
        self.config_obj.fold = fold

    def change_config_rel_op(self, train=True):
        """Changes to training or inference for the relational model."""
        self.config_obj.infer = not train
