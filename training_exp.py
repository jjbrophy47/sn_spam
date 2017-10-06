import os
import sys
from app import run


class Training_Experiment:

    def __init__(self, config_obj, runner_obj, modified=True, pseudo=True):
        self.config_obj = config_obj
        self.runner_obj = runner_obj
        self.config_obj.modified = modified
        self.config_obj.pseudo = pseudo

    def divide_data_into_subsets(self, growth_factor=2,
            train_start_size=100, train_split=0.7, val_split=0.3):
        """Changes to: start, train_size, val_size, fold.
                Use start and end in config as test set."""
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

    def change_config_parameters(self, start, train_size, val_size, fold):
        self.config_obj.start = start
        self.config_obj.train_size = train_size
        self.config_obj.val_size = val_size
        self.config_obj.fold = fold

    def change_config_relational_operation(self, train=True):
        self.config_obj.infer = not train

    def run_experiment(self):
        val_df, test_df = self.runner_obj.run_independent()
        self.runner_obj.run_purity(test_df)
        self.change_config_relational_operation(train=True)
        self.runner_obj.run_relational(val_df, test_df)
        self.change_config_relational_operation(train=False)
        self.runner_obj.run_relational(val_df, test_df)
        self.runner_obj.run_evaluation(test_df)


if __name__ == '__main__':
    """Sets up the project and runs the experiment."""
    args = sys.argv
    this_dir = os.path.abspath(os.getcwd())
    app_dir, ind_dir, rel_dir, ana_dir = run.directories(this_dir)
    runner_obj, config_obj = run.init_dependencies()

    config_obj.set_directories(app_dir, ind_dir, rel_dir, ana_dir)
    config_obj.set_options(args)
    config_obj.parse_config()
    run.global_settings(config_obj)
    # runner_obj.compile_reasoning_engine()

    se = Training_Experiment(config_obj, runner_obj)
    subsets = se.divide_data_into_subsets(growth_factor=2,
            train_start_size=1000)

    for start, train_size, val_size, fold in subsets:
        print(start, train_size, val_size, fold)
        # se.change_config_parameters(start, train_size, val_size, fold)
        # se.run_experiment()
