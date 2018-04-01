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
from experiments.relations_exp import Relations_Experiment
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

    connections_obj = Connections(util_obj)
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
    relational_obj = Relational(connections_obj, config_obj, psl_obj,
                                tuffy_obj, mrf_obj, util_obj)

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


def add_args():
    description = 'Spam detection for online social networks'
    parser = argparse.ArgumentParser(description=description)

    # high level args
    parser.add_argument('-r', '--run', help='Run detection engine',
                        action='store_true')
    parser.add_argument('--ablation', help='Run ablation experiment',
                        action='store_true')
    parser.add_argument('--learning', help='Run learning curves experiment',
                        action='store_true')
    parser.add_argument('--relations', help='Run relations experiment',
                        action='store_true')
    parser.add_argument('--stacking', help='Run stacking experiment',
                        action='store_true')
    parser.add_argument('--subsets', help='Run subsets experiment',
                        action='store_true')

    # general args that overlap among different APIs
    parser.add_argument('-d', '--domain', default='twitter',
                        help='social network, default: %(default)s')
    parser.add_argument('-s', '--start', default=0, type=int,
                        help='data range start, default: %(default)s')
    parser.add_argument('-e', '--end', default=1000, type=int,
                        help='data range end, default: %(default)s')
    parser.add_argument('--engine', default=None,
                        help='relational framework, default: %(default)s')
    parser.add_argument('--clf', default='lgb',
                        help='classifier, default: %(default)s')
    parser.add_argument('--ngrams', action='store_true',
                        help='use ngrams, default: %(default)s')
    parser.add_argument('--stacks', default=0, type=int,
                        help='number of stacks, default: %(default)s')
    parser.add_argument('--data', default='both',
                        help='rel, ind, or both, default: %(default)s')
    parser.add_argument('--train_size', default=0.8, type=float,
                        help='train size, default: %(default)s')
    parser.add_argument('--val_size', default=0.1, type=float,
                        help='val size, default: %(default)s')
    parser.add_argument('--param_search', default='single',
                        help='parameter search, default: %(default)s')
    parser.add_argument('--tune_size', default=0.2, type=float,
                        help='tuning size, default: %(default)s')
    parser.add_argument('--separate_relations', action='store_true',
                        help='break set relations, default: %(default)s')
    parser.add_argument('--eval', default='cc',
                        help='type of testing, default: %(default)s')
    parser.add_argument('--rels', nargs='*',
                        help='relations to exploit, default: %(default)s')

    # experiment specific args
    parser.add_argument('--train_sizes', nargs='*',
                        help='training sizes, default: %(default)s')
    parser.add_argument('--start_stack', default=0, type=int,
                        help='beginning stack number, default: %(default)s')
    parser.add_argument('--end_stack', default=4, type=int,
                        help='ending stack number, default: %(default)s')
    return parser


def parse_args(parser):
    p = {}
    args = parser.parse_args()

    p['domain'] = args.domain
    p['start'] = args.start
    p['end'] = args.end
    p['engine'] = args.engine
    p['clf'] = args.clf
    p['ngrams'] = args.ngrams
    p['stacks'] = args.stacks
    p['data'] = args.data
    p['train_size'] = args.train_size
    p['val_size'] = args.val_size
    p['tune_size'] = args.tune_size
    p['param_search'] = args.param_search
    p['separate_relations'] = args.separate_relations
    p['eval'] = args.eval
    p['relations'] = args.rels if args.rels is not None else []
    p['train_sizes'] = args.train_sizes if args.train_sizes is not None else []
    p['start_stack'] = args.start_stack
    p['end_stack'] = args.end_stack

    return args, p


def main():
    parser = add_args()
    args, p = parse_args(parser)

    this_dir = os.path.abspath(os.getcwd())
    app_dir, ind_dir, rel_dir, ana_dir = directories(this_dir)
    config_obj, app_obj = init_dependencies()

    global_settings(config_obj)
    config_obj.set_directories(app_dir, ind_dir, rel_dir, ana_dir)

    if args.run:
        app_obj.run(domain=p['domain'], start=p['start'], end=p['end'],
                    engine=p['engine'], clf=p['clf'], ngrams=p['ngrams'],
                    stacking=p['stacks'], data=p['data'],
                    train_size=p['train_size'], val_size=p['val_size'],
                    relations=p['relations'],
                    separate_relations=p['separate_relations'],
                    evaluation=p['eval'], param_search=p['param_search'],
                    tune_size=p['tune_size'])

    elif args.ablation:
        le = Ablation_Experiment(config_obj, app_obj)
        le.run_experiment(start=p['start'], end=p['end'], domain=p['domain'],
                          featuresets=['content', 'ngrams', 'sequential'])

    elif args.learning:
        le = Learning_Experiment(config_obj, app_obj)
        le.run_experiment(test_start=p['start'], test_end=p['end'],
                          train_sizes=p['train_sizes'], domain=p['domain'],
                          start_fold=0)

    elif args.relations:
        le = Relations_Experiment(config_obj, app_obj)
        le.run_experiment(start=p['start'], end=p['end'], domain=p['domain'],
                          relationsets=p['relations'])

    elif args.stacking:
        se = Stacking_Experiment(config_obj, app_obj)
        se.run_experiment(domain=p['domain'], start=p['start'], end=p['end'],
                          start_stack=p['start_stack'],
                          end_stack=p['end_stack'], relations=p['relations'])

    elif args.subsets:
        se = Subsets_Experiment(config_obj, app_obj)
        se.run_experiment(domain='twitter', start=0, end=1000, subsets=5,
                          data='both')
