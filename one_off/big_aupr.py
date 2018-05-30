import os
import argparse
import util as ut
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve


def compute_big_aupr(start_fold=0, num_folds=5, domain='twitter',
                     models=['ind']):
    ind_data_dir = 'independent/data/' + domain + '/'
    ind_preds_dir = 'independent/output/' + domain + '/predictions/'
    rel_preds_dir = 'relational/output/' + domain + '/predictions/'

    lines = {'ind': 'b-', 'mrf': 'g--', 'psl': 'm-.'}
    inds, mrfs, psls = [], [], []
    preds = []

    for model in models:
        preds.append(model + '_pred')
    preds = list(zip(models, preds))

    s = '%s: reading model preds from fold %d to %d:'
    ut.out(s % (domain, start_fold, num_folds - 1), 0)

    for i in range(start_fold, start_fold + num_folds):
        ut.out('reading preds for fold %d...' % i)

        if 'ind' in models:
            fname = ind_preds_dir + 'test_' + str(i) + '_preds.csv'
            assert os.path.exists(fname)
            inds.append(pd.read_csv(fname))

        if 'mrf' in models:
            fname = rel_preds_dir + 'mrf_preds_' + str(i) + '.csv'
            assert os.path.exists(fname)
            mrfs.append(pd.read_csv(fname))

        if 'psl' in models:
            fname = rel_preds_dir + 'psl_preds_' + str(i) + '.csv'
            assert os.path.exists(fname)
            psls.append(pd.read_csv(fname))

    ut.out('reading true labels...')
    df = pd.read_csv(ind_data_dir + 'comments.csv')
    df = df[['com_id', 'label']]

    if 'ind' in models:
        ind_df = pd.concat(inds)
        df = df.merge(ind_df)

    if 'mrf' in models:
        mrf_df = pd.concat(mrfs)
        assert set(ind_df['com_id']) == set(mrf_df['com_id'])
        df = df.merge(mrf_df)

    if 'psl' in models:
        psl_df = pd.concat(psls)
        assert set(ind_df['com_id']) == set(psl_df['com_id'])
        df = df.merge(psl_df)

    # compute combined test set aupr
    for i, (model, pred) in enumerate(preds):
        aupr = average_precision_score(df['label'], df[pred])
        ut.out('%s aupr: %.4f' % (model, aupr))
        p, r, t = precision_recall_curve(df['label'], df[pred])
        save = True if i == len(preds) - 1 else False
        ut.plot_pr_curve(model, p, r, aupr, save=save, domain=domain,
                         line=lines[model], show_legend=True)
    ut.out()


if __name__ == '__main__':
    description = 'Script to merge and compute subset predictions scores'
    parser = argparse.ArgumentParser(description=description, prog='big_aupr')

    parser.add_argument('-d', metavar='DOMAIN',
                        help='domain, default: %(default)s')
    parser.add_argument('--start_fold', metavar='NUM', type=int,
                        help='first subset, default: %(default)s')
    parser.add_argument('--num_folds', metavar='NUM', type=int,
                        help='number of subsets, default: %(default)s')
    parser.add_argument('--models', nargs='*', metavar='MODEL',
                        help='list of models, default: %(default)s')

    args = parser.parse_args()
    domain = args.d
    start = args.start_fold
    folds = args.num_folds
    models = args.models if args.models is not None else ['ind']

    compute_big_aupr(start_fold=start, num_folds=folds, domain=domain,
                     models=models)
