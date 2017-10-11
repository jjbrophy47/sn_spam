"""
Module to test one test set one time.
"""


class Single_Experiment:
    """Handles all operations to run the independent and relational models."""

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
    def run_experiment(self):
        """Runs the independent and relational models."""
        self.single_run()

    # private
    def single_run(self):
        """Operations to run the independent model, train the relational model
                from those predictions, and then do joint prediction using
                the relational model on the test set."""
        val_df, test_df = self.runner_obj.run_independent()
        self.change_config_rel_op(train=True)
        self.runner_obj.run_relational(val_df, test_df)
        self.change_config_rel_op(train=False)
        self.runner_obj.run_relational(val_df, test_df)
        self.runner_obj.run_evaluation(test_df)

    def change_config_rel_op(self, train=True):
        """Changes to training or inference for the relational model.
        train: boolean to train relational model if True, do inference
                otherwise."""
        self.config_obj.infer = not train
