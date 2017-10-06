import os
import sys
from app import run


class Subsets_Experiment:

    def __init__(self, config_obj, runner_obj):
        self.config_obj = config_obj
        self.runner_obj = runner_obj
        self.config_obj.modified = True

    def divide_data_into_subsets(self, num_subsets=10):
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

    def change_config_parameters(self, start, end, fold):
        self.config_obj.start = start
        self.config_obj.end = end
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
    runner_obj.compile_reasoning_engine()

    se = Subsets_Experiment(config_obj, runner_obj)
    subsets = se.divide_data_into_subsets(num_subsets=10)

    for start, end, fold in subsets:
        print(start, end, fold)
        se.change_config_parameters(start, end, fold)
        se.run_experiment()
