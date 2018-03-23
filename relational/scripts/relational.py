"""
This module generates predicates for and runs the relational model.
"""
import os
import math
import pandas as pd
from operator import itemgetter


class Relational:

    def __init__(self, connections_obj, config_obj, psl_obj, tuffy_obj,
                 mrf_obj, util_obj):
        self.conns_obj = connections_obj
        self.config_obj = config_obj
        self.psl_obj = psl_obj
        self.tuffy_obj = tuffy_obj
        self.mrf_obj = mrf_obj
        self.util_obj = util_obj

    # public
    def compile_reasoning_engine(self):
        """Uses the psl module to compile the PSL and groovy scripts."""
        psl_f, _, _, _, _, _, _, _ = self._folders()
        self.psl_obj.compile(psl_f)

    def main(self, val_df, test_df):
        """Sets up file structure, merges predictions, and runs the model.
        val_df: validation dataframe.
        test_df: testing dataframe."""
        f = self._folders()
        psl_f, psl_d_f, tuffy_f, mrf_f, folds_f, pred_f, rel_pred_f, sts_f = f

        val_df, test_df = self._check_dataframes(val_df, test_df, folds_f)
        val_df = self._merge_ind_preds(val_df, 'val', pred_f)
        test_df = self._merge_ind_preds(test_df, 'test', pred_f)

        self._run_relational_model(val_df, test_df, psl_f, psl_d_f, tuffy_f,
                                   mrf_f, rel_pred_f)

    # private
    def _folders(self):
        """Returns absolute path directories."""
        ind_dir = self.config_obj.ind_dir
        rel_dir = self.config_obj.rel_dir
        domain = self.config_obj.domain

        psl_f = rel_dir + 'psl/'
        psl_data_f = psl_f + 'data/' + domain + '/'
        tuffy_f = rel_dir + 'tuffy/'
        mrf_f = rel_dir + 'mrf/'
        folds_f = ind_dir + 'data/' + domain + '/folds/'
        pred_f = ind_dir + 'output/' + domain + '/predictions/'
        rel_pred_f = rel_dir + 'output/' + domain + '/predictions/'
        status_f = rel_dir + 'output/' + domain + '/status/'
        if not os.path.exists(psl_data_f):
            os.makedirs(psl_data_f)
        if not os.path.exists(status_f):
            os.makedirs(status_f)
        folders = (psl_f, psl_data_f, tuffy_f, mrf_f, folds_f, pred_f,
                   rel_pred_f, status_f)
        return folders

    def _check_dataframes(self, val_df, test_df, folds_f):
        fold = self.config_obj.fold

        if val_df is None or test_df is None:
            val_df = pd.read_csv(folds_f + 'val_' + fold + '.csv')
            test_df = pd.read_csv(folds_f + 'test_' + fold + '.csv')
        return val_df, test_df

    def _merge_ind_preds(self, df, dset, ind_pred_f):
        """Merge the independent model predictions with the relational ones.
        df: dataframe containing comments.
        dset: dataset (e.g. 'val', 'test')
        ind_pred_f: folder where the independent predictions are stored.
        Returns merged dataframe."""
        fold = self.config_obj.fold
        preds_df = pd.read_csv(ind_pred_f + dset + '_' + fold + '_preds.csv')
        df = df.merge(preds_df)
        return df

    def _run_psl(self, val_df, test_df, psl_f, psl_d, rel_d):
        max_size = 40000

        if not self.config_obj.infer:  # train
            self.psl_obj.gen_predicates(val_df, 'val', psl_d)
            self.psl_obj.gen_model(psl_d)
            self.psl_obj.network_size(psl_d)
            self.psl_obj.run(psl_f)

        else:  # test
            self.psl_obj.gen_predicates(test_df, 'test', psl_d)
            size = self.psl_obj.network_size(psl_d)

            if size >= max_size:
                self.util_obj.out('size > %d...' % max_size)
                relations = self.config_obj.relations
                subgraphs = self.conns_obj.find_subgraphs(test_df, relations)
                subgraphs = self.conns_obj.consolidate(subgraphs, max_size)

                for i, (ids, rels, edges) in enumerate(subgraphs):
                    s = 'reasoning over sg_%d with %d msgs and %d edges...'
                    self.util_obj.out(s % (i, len(ids), edges))
                    test_sg_df = test_df[test_df['com_id'].isin(ids)]
                    self.psl_obj.gen_predicates(test_sg_df, 'test', psl_d, i)
                    self.psl_obj.run(psl_f, i)
                self.psl_obj.combine_predictions(len(subgraphs), rel_d)
            else:
                self.psl_obj.run(psl_f)

    def run_tuffy(self, val_df, test_df, tuffy_f, fw=None):
        self.tuffy_obj.clear_data(tuffy_f)

        self.util_obj.start('\nbuilding predicates...', fw=fw)
        self.tuffy_obj.gen_predicates(val_df, 'val', tuffy_f)
        self.tuffy_obj.gen_predicates(test_df, 'test', tuffy_f)
        self.util_obj.end('\n\ttime: ', fw=fw)
        self.tuffy_obj.run(tuffy_f)
        pred_df = self.tuffy_obj.parse_output(tuffy_f)
        self.tuffy_obj.evaluate(test_df, pred_df)

    def _run_mrf(self, val_df, test_df, mrf_f, rel_pred_f, fw=None):
        ut = self.util_obj
        max_size = 7500

        # train
        ep_scores = []
        epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        ut.out('tuning epsilon: %s...' % str(epsilons))
        for ep in epsilons:
            ut.out('%.2f...' % ep)
            md, rd = self.mrf_obj.gen_mn(val_df, 'val', mrf_f, ep)
            self.mrf_obj.run(mrf_f, dset='val')
            preds_df = self.mrf_obj.process_marginals(md, mrf_f,
                                                      dset='val',
                                                      pred_dir=rel_pred_f)
            ep_score = self.mrf_obj.compute_aupr(preds_df, val_df)
            ep_scores.append((ep, ep_score))
        b_ep = max(ep_scores, key=itemgetter(1))[0]
        ut.out('best epsilon: %.2f' % b_ep)

        # test
        ut.out('test set inference...')
        md, rd = self.mrf_obj.gen_mn(test_df, 'test', mrf_f, b_ep)
        size = self.mrf_obj.network_size(md, rd)

        if size > max_size:
            self.util_obj.out('size > %d, finding subgraphs...' % max_size)
            relations = self.config_obj.relations
            subgraphs = self.conns_obj.find_subgraphs(test_df, relations)
            subgraphs = self.conns_obj.consolidate(subgraphs, max_size)

            dfs = []
            for i, (ids, rels, edges) in enumerate(subgraphs):
                s = 'reasoning over sg_%d with %d msgs and %d edges...'
                self.util_obj.out(s % (i, len(ids), edges))
                test_sg_df = test_df[test_df['com_id'].isin(ids)]
                md, rd = self.mrf_obj.gen_mn(test_sg_df, 'test', mrf_f, b_ep)
                self.mrf_obj.run(mrf_f, dset='test')
                df = self.mrf_obj.process_marginals(md, mrf_f,
                                                    dset='test',
                                                    pred_dir=rel_pred_f)
                dfs.append(df)
            df = pd.concat(dfs)
            fold = self.config_obj.fold
            df.to_csv(rel_pred_f + 'mrf_preds_' + fold + '.csv', index=None)

        else:
            self.mrf_obj.run(mrf_f, dset='test')
            self.mrf_obj.process_marginals(md, mrf_f, dset='test',
                                           pred_dir=rel_pred_f)

    def _run_relational_model(self, val_df, test_df, psl_f, psl_data_f,
                              tuffy_f, mrf_f, rel_pred_f,
                              transform='logistic'):
        val_df = self._transform_priors(val_df, transform=transform)
        test_df = self._transform_priors(test_df, transform=transform)

        if self.config_obj.engine == 'psl':
            self._run_psl(val_df, test_df, psl_f, psl_data_f, rel_pred_f)
        elif self.config_obj.engine == 'tuffy':
            self._run_tuffy(val_df, test_df, tuffy_f)
        elif self.config_obj.engine == 'mrf':
            self._run_mrf(val_df, test_df, mrf_f, rel_pred_f)

    def _transform_priors(self, df, col='ind_pred', transform='logit'):
        clf = self.config_obj.classifier
        eng = self.config_obj.engine

        if clf != 'lr' and eng in ['mrf', 'all']:

            if transform is not None:
                if transform == 'e':
                    scale = self._transform_e
                elif transform == 'logit':
                    scale = self._transform_logit
                elif transform == 'logistic':
                    scale = self._transform_logistic

                df['ind_pred'] = df['ind_pred'].apply(scale)
        return df

    def _transform_e(self, x):
        result = x

        if x == 0:
            result = 0
        elif x == 1:
            result == 1
        else:
            result = math.exp(x)
        return result

    def _transform_logistic(self, x, alpha=2):
        result = x

        if x == 0:
            result = 0
        elif x == 1:
            result == 1
        else:
            result = (x ** alpha) / (x + ((1 - x) ** alpha))
        return result

    def _transform_logit(self, x):
        result = x

        if x == 0:
            result = 0
        elif x == 1:
            result == 1
        else:
            result = math.log(x / (1 - x))
        return result
