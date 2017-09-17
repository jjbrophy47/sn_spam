import os
import sys
import warnings
import pandas as pd
from app.runner import Runner
from app.config import Config
from independent.scripts.independent import Independent
from independent.scripts.classification import Classification
from independent.scripts.content_features import ContentFeatures
from independent.scripts.graph_features import GraphFeatures
from independent.scripts.relational_features import RelationalFeatures
from relational.scripts.comments import Comments
from relational.scripts.generator import Generator
from relational.scripts.pred_builder import PredicateBuilder
from relational.scripts.psl import PSL
from relational.scripts.relational import Relational
from relational.scripts.tuffy import Tuffy
from analysis.analysis import Analysis
from analysis.connections import Connections
from analysis.label import Label
from analysis.purity import Purity
from analysis.evaluation import Evaluation
from analysis.interpretability import Interpretability
from analysis.util import Util


def directories(this_dir):
    """Sets up absolute directories.
    this_dir: current working directory.
    Returns absolute directories to the config, independent, relational, and
            analysis packages."""
    app_dir = this_dir + '/app/'
    ind_dir = this_dir + '/independent/'
    rel_dir = this_dir + '/relational/'
    ana_dir = this_dir + '/analysis/'
    return app_dir, ind_dir, rel_dir, ana_dir


def init_dependencies():
    """Initializes all dependencies. Returns the Runner and Config objects."""
    config_obj = Config()
    util_obj = Util()

    content_features_obj = ContentFeatures(config_obj, util_obj)
    graph_features_obj = GraphFeatures(config_obj, util_obj)
    relational_features_obj = RelationalFeatures(config_obj, util_obj)
    classify_obj = Classification(config_obj, content_features_obj,
            graph_features_obj, relational_features_obj, util_obj)
    independent_obj = Independent(config_obj, classify_obj, util_obj)

    generator_obj = Generator()
    comments_obj = Comments(config_obj, util_obj)
    pred_builder_obj = PredicateBuilder(config_obj, comments_obj,
        generator_obj, util_obj)
    psl_obj = PSL(config_obj, pred_builder_obj)
    tuffy_obj = Tuffy(config_obj, pred_builder_obj)
    relational_obj = Relational(config_obj, psl_obj, tuffy_obj, util_obj)

    connections_obj = Connections()
    label_obj = Label(config_obj, generator_obj, util_obj)
    purity_obj = Purity(config_obj, generator_obj, util_obj)
    evaluate_obj = Evaluation(config_obj, generator_obj, util_obj)
    interpret_obj = Interpretability(config_obj, connections_obj,
            generator_obj, pred_builder_obj, util_obj)
    analysis_obj = Analysis(config_obj, label_obj, purity_obj, evaluate_obj,
            interpret_obj, util_obj)

    runner_obj = Runner(independent_obj, relational_obj, analysis_obj)
    return runner_obj, config_obj


def global_settings(config_obj):
    """Settings used throughout the application.
    config_obj: user settings."""
    pd.options.mode.chained_assignment = None
    warnings.filterwarnings(action="ignore", module="scipy",
            message="^internal gelsd")
    if os.isatty(sys.stdin.fileno()):
        rows, columns = os.popen('stty size', 'r').read().split()
        pd.set_option('display.width', int(columns))
        config_obj.set_display(True)


def main():
    """Sets up the project and runs the application."""
    args = sys.argv
    this_dir = os.path.abspath(os.getcwd())
    app_dir, ind_dir, rel_dir, ana_dir = directories(this_dir)
    runner_obj, config_obj = init_dependencies()

    config_obj.set_directories(app_dir, ind_dir, rel_dir, ana_dir)
    config_obj.set_options(args)
    config_obj.parse_config()
    global_settings(config_obj)

    val_df, test_df = None, None

    if '-r' in args or '-x' in args:
        runner_obj.compile_reasoning_engine()

    if '-l' in args:
        runner_obj.run_label()
        print('done, exiting...')
        exit(0)

    if '-i' in args:
        val_df, test_df = runner_obj.run_independent()

    if '-p' in args:
        runner_obj.run_purity(test_df)

    if '-r' in args:
        runner_obj.run_relational(val_df, test_df)

    if '-e' in args:
        runner_obj.run_evaluation(test_df)

    if '-x' in args:
        runner_obj.run_explanation(test_df)
