import os
import sys
import argparse
import warnings
import pandas as pd
from app.config import Config
from app.data import Data
from app.app import App
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
from relational.scripts.mrf import MRF
from analysis.analysis import Analysis
from analysis.connections import Connections
from analysis.label import Label
from analysis.purity import Purity
from analysis.evaluation import Evaluation
from analysis.interpretability import Interpretability
from analysis.util import Util
from experiments.learning_exp import Learning_Experiment
from experiments.stacking_exp import Stacking_Experiment
from experiments.subsets_exp import Subsets_Experiment
from experiments.ablation_exp import Ablation_Experiment


def directories(this_dir):
    app_dir = this_dir + '/app/'
    ind_dir = this_dir + '/independent/'
    rel_dir = this_dir + '/relational/'
    ana_dir = this_dir + '/analysis/'
    return app_dir, ind_dir, rel_dir, ana_dir


def init_dependencies():
    config_obj = Config()
    util_obj = Util()

    generator_obj = Generator(util_obj)
    data_obj = Data(generator_obj, util_obj)

    content_features_obj = ContentFeatures(config_obj, util_obj)
    graph_features_obj = GraphFeatures(config_obj, util_obj)
    relational_features_obj = RelationalFeatures(config_obj, util_obj)
    classify_obj = Classification(config_obj, content_features_obj,
                                  graph_features_obj, relational_features_obj,
                                  util_obj)
    independent_obj = Independent(config_obj, classify_obj, generator_obj,
                                  util_obj)

    comments_obj = Comments(config_obj, util_obj)
    pred_builder_obj = PredicateBuilder(config_obj, comments_obj,
                                        generator_obj, util_obj)
    psl_obj = PSL(config_obj, pred_builder_obj, util_obj)
    tuffy_obj = Tuffy(config_obj, pred_builder_obj, util_obj)
    mrf_obj = MRF(config_obj, util_obj, generator_obj)
    relational_obj = Relational(config_obj, psl_obj, tuffy_obj, mrf_obj,
                                util_obj)

    connections_obj = Connections()
    label_obj = Label(config_obj, generator_obj, util_obj)
    purity_obj = Purity(config_obj, generator_obj, util_obj)
    evaluate_obj = Evaluation(config_obj, generator_obj, util_obj)
    interpret_obj = Interpretability(config_obj, connections_obj,
                                     generator_obj, pred_builder_obj, util_obj)
    analysis_obj = Analysis(config_obj, label_obj, purity_obj, evaluate_obj,
                            interpret_obj, util_obj)

    app_obj = App(config_obj, data_obj, independent_obj, relational_obj,
                  analysis_obj, util_obj)
    return config_obj, app_obj


def global_settings(config_obj):
    pd.options.mode.chained_assignment = None
    warnings.filterwarnings(action="ignore", module="scipy",
                            message="^internal gelsd")
    if os.isatty(sys.stdin.fileno()):
        rows, columns = os.popen('stty size', 'r').read().split()
        pd.set_option('display.width', int(columns))
        config_obj.set_display(True)


def main():
    description = 'Spam detection for online social networks'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-r', '--run', help='Run detection engine',
                        action='store_true')
    parser.add_argument('--ablation', help='Run ablation experiment',
                        action='store_true')
    parser.add_argument('--learning', help='Run learning curves experiment',
                        action='store_true')
    parser.add_argument('--stacking', help='Run stacking experiment',
                        action='store_true')
    parser.add_argument('--subsets', help='Run subsets experiment',
                        action='store_true')
    args = parser.parse_args()

    this_dir = os.path.abspath(os.getcwd())
    app_dir, ind_dir, rel_dir, ana_dir = directories(this_dir)
    config_obj, app_obj = init_dependencies()

    global_settings(config_obj)
    config_obj.set_directories(app_dir, ind_dir, rel_dir, ana_dir)

    if args.run:
        app_obj.run(domain='adclicks', start=0, end=1000000,
                    engine=None, param_search='single',
                    clf='xgb', ngrams=False, stacking=0, data='both',
                    train_size=0.9, val_size=0, tune_size=0.15,
                    relations=[], evaluation='cc')

        # app_obj.run(domain='twitter', start=0, end=10000,
        #             engine='psl', clf='lr', ngrams=False, stacking=1,
        #             data='both', train_size=0.7, val_size=0.15,
        #             relations=['inhash', 'posts', 'intext'],
        #             separate_relations=True, evaluation='cc')

    elif args.ablation:
        le = Ablation_Experiment(config_obj, app_obj)
        le.run_experiment(start=0, end=20000, domain='adclicks',
                          featuresets=['base', 'content', 'sequential'])

    elif args.learning:
        train_sizes = [100000, 200000, 400000, 800000, 1600000, 3200000,
                       6400000, 10000000]
        le = Learning_Experiment(config_obj, app_obj)
        le.run_experiment(test_start=10000000, test_end=11000000,
                          train_sizes=train_sizes, domain='adclicks',
                          start_fold=0)

    elif args.stacking:
        se = Stacking_Experiment(config_obj, app_obj)
        se.run_experiment(domain='adclicks', start=0, end=2000000,
                          start_stack=2, end_stack=7,
                          relations=['hasip', 'hasos', 'hasdevice',
                                     'inchannel', 'inapp'])

    elif args.subsets:
        se = Subsets_Experiment(config_obj, app_obj)
        se.run_experiment(domain='twitter', start=0, end=1000, subsets=5,
                          data='both')
