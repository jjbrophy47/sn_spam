import os
import sys
from app import run


class Robust_Experiment:

    def __init__(self, config_obj, runner_obj, modified=True, pseudo=True):
        self.config_obj = config_obj
        self.runner_obj = runner_obj
        self.config_obj.modified = modified
        self.config_obj.pseudo = pseudo

    def change_config_parameters(self, alter_user_ids=False):
        self.config_obj.alter_user_ids = alter_user_ids
        self.config_obj.fold = str(int(self.config_obj.fold) + 1)

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

    re = Robust_Experiment(config_obj, runner_obj)
    re.run_experiment()
    re.change_config_parameters(alter_user_ids=True)
    re.run_experiment()
