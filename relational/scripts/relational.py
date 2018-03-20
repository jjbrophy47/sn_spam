"""
This module generates predicates for and runs the relational model.
"""
import os
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

    # def open_status_writer(self, status_f, mode='w'):
    #     fold = self.config_obj.fold
    #     operation = 'infer' if self.config_obj.infer else 'train'
    #     fname = status_f + operation + '_' + fold + '.txt'
    #     f = self.util_obj.open_writer(fname, mode)
    #     return f

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
                relations = self.config_obj.relations
                self.util_obj.out('size > %d, finding subgraphs...' % max_size)
                subgraphs = self.conns_obj.find_subgraphs(test_df, relations)
                subgraphs = self.conns_obj.consolidate(subgraphs, max_size)

                for i, (ids, rels) in enumerate(subgraphs):
                    self.util_obj.out('reasoning sg %d: %d' % (i, len(ids)))
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

        ep_scores = []
        epsilons = [0.1, 0.2, 0.3]
        ut.out('tuning epsilon: %s' % str(epsilons))
        for ep in epsilons:
            ut.out('%f...' % ep)
            msgs_dict = self.mrf_obj.gen_mn(val_df, 'val', mrf_f, ep, fw=fw)
            self.mrf_obj.run(mrf_f, dset='val')
            preds_df = self.mrf_obj.process_marginals(msgs_dict, mrf_f,
                                                      dset='val',
                                                      pred_dir=rel_pred_f)
            ep_score = self.mrf_obj.compute_aupr(preds_df, val_df)
            ep_scores.append((ep, ep_score))
        best_ep = max(ep_scores, key=itemgetter(1))[0]
        ut.out('best epsilon: %f' % best_ep)

        # self.mrf_obj.clear_data(mrf_f)
        ut.out('doing inference on test set...')
        msgs_dict = self.mrf_obj.gen_mn(test_df, 'test', mrf_f, best_ep, fw=fw)
        self.mrf_obj.run(mrf_f, dset='test')
        self.mrf_obj.process_marginals(msgs_dict, mrf_f, dset='test',
                                       pred_dir=rel_pred_f)

    def _run_relational_model(self, val_df, test_df, psl_f, psl_data_f,
                              tuffy_f, mrf_f, rel_pred_f):
        if self.config_obj.engine == 'psl':
            self._run_psl(val_df, test_df, psl_f, psl_data_f, rel_pred_f)
        elif self.config_obj.engine == 'tuffy':
            self._run_tuffy(val_df, test_df, tuffy_f)
        elif self.config_obj.engine == 'mrf':
            self._run_mrf(val_df, test_df, mrf_f, rel_pred_f)
