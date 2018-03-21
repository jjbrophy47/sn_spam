"""
Module that classifies data using an independent model.
"""
import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.sparse import csr_matrix


class Classification:
    def __init__(self, config_obj, content_features_obj, graph_features_obj,
                 relational_features_obj, util_obj):
        self.config_obj = config_obj
        self.cf_obj = content_features_obj
        self.gf_obj = graph_features_obj
        self.rf_obj = relational_features_obj
        self.util_obj = util_obj

    # public
    def main(self, train_df, test_df, dset='test'):
        """Constructs paths, merges data, converts, and processes data to be
        read by the independent model.
        train_df: original training comments dataframe.
        test_df: original testing comments dataframe.
        dset: datatset to test (e.g. 'val', 'test')."""
        stacking = self.config_obj.stacking

        if stacking > 0:
            self.do_stacking(train_df, test_df, dset, stacking=stacking)
        else:
            self.do_normal(train_df, test_df, dset)

    # private
    def do_stacking(self, train_df, test_df, dset='test', stacking=1):
        self.util_obj.out('stacking with %d stack(s)...' % stacking)
        fold = self.config_obj.fold
        clf = self.config_obj.classifier
        ps = self.config_obj.param_search
        ts = self.config_obj.tune_size

        image_f, pred_f, model_f = self.file_folders()
        trains = self.split_training_data(train_df, splits=stacking + 1)
        test_df = test_df.copy()

        for i in range(len(trains)):
            t = 'train_' + str(i)
            d_tr, cv = self.build_and_merge(trains[i], 'train', t=t)
            learner = self.util_obj.train(d_tr, clf, ps, ts)

            for j in range(i + 1, len(trains)):
                t = 'train_' + str(j)
                d_te, _ = self.build_and_merge(trains[j], 'test', cv=cv, t=t)
                te_preds, ids = self.util_obj.test(d_te, learner, 1)
                trains[j] = self.append_preds(trains[j], te_preds, ids)

            d_te, _ = self.build_and_merge(test_df, 'test', cv=cv, t='test')
            te_preds, ids = self.util_obj.test(d_te, learner, 1)
            test_df = self.append_preds(test_df, te_preds, ids)

        self.util_obj.evaluate(d_te, te_preds)
        self.util_obj.save_preds(te_preds, ids, fold, pred_f, dset)

        if not self.config_obj.ngrams:
            _, _, _, feats = d_te
            self.util_obj.plot_features(learner, clf, feats, image_f + 'a')

    def do_normal(self, train_df, test_df, dset='test'):
        self.util_obj.out('normal...')
        fold = self.config_obj.fold
        clf = self.config_obj.classifier
        ps = self.config_obj.param_search
        ts = self.config_obj.tune_size

        image_f, pred_f, model_f = self.file_folders()

        # train base learner using training set.
        d_tr, cv = self.build_and_merge(train_df, 'train', t='train')
        learner = self.util_obj.train(d_tr, clf, ps, ts)

        # test learner on test set.
        d_te, _ = self.build_and_merge(test_df, 'test', t='test')
        y_score, ids = self.util_obj.test(d_te, learner)
        self.util_obj.evaluate(d_te, y_score)
        self.util_obj.save_preds(y_score, ids, fold, pred_f, dset)

        if not self.config_obj.ngrams:
            _, _, _, feats = d_te
            self.util_obj.plot_features(learner, clf, feats, image_f + 'a')

    def file_folders(self):
        ind_dir = self.config_obj.ind_dir
        domain = self.config_obj.domain

        out_f = ind_dir + 'output/' + domain
        image_f = out_f + '/images/'
        pred_f = out_f + '/predictions/'
        model_f = out_f + '/models/'

        if not os.path.exists(image_f):
            os.makedirs(image_f)
        if not os.path.exists(pred_f):
            os.makedirs(pred_f)
        if not os.path.exists(model_f):
            os.makedirs(model_f)
        return image_f, pred_f, model_f

    def merge(self, coms_df, c_df, g_df, r_df):
        feats_df = coms_df.merge(c_df, on='com_id', how='left')
        feats_df = feats_df.merge(g_df, on='com_id', how='left')
        feats_df = feats_df.merge(r_df, on='com_id', how='left')
        return feats_df

    def drop_columns(self, feats_df, feats_list):
        return feats_df[feats_list]

    def dataframe_to_matrix(self, feats_df):
        return csr_matrix(feats_df.as_matrix())

    def stack_matrices(self, feats_m, c_csr):
        stack = [feats_m]
        if c_csr is not None:
            stack.append(c_csr)
        return hstack(stack).tocsr()

    def extract_ids_and_labels(self, df):
        ids = df['com_id'].values
        labels = df['label'].values if 'label' in list(df) else None
        return ids, labels

    def prepare(self, df, c_m, c_df, g_df, r_df, feats_list):
        self.util_obj.out('merging features...')

        feats_df = self.merge(df, c_df, g_df, r_df)
        feats_df = self.drop_columns(feats_df, feats_list)
        # self.util_obj.out(str(feats_df.head(5)))

        feats_m = self.dataframe_to_matrix(feats_df)
        x = self.stack_matrices(feats_m, c_m)
        self.util_obj.out(str(x.shape), 0)

        ids, y = self.extract_ids_and_labels(df)
        return x, y, ids

    def build_and_merge(self, df, dset, cv=None, t='te1'):
        self.util_obj.out('\nbuilding features for %s:' % t)
        m, c_df, c_feats, cv = self.cf_obj.build(df, dset, cv=cv)
        g_df, g_feats = self.gf_obj.build(df)
        r_df, r_feats = self.rf_obj.build(df, dset)
        feats = c_feats + g_feats + r_feats
        x, y, ids = self.prepare(df, m, c_df, g_df, r_df, feats)
        return (x, y, ids, feats), cv

    def append_preds(self, test_df, test_probs, id_te):
        if 'noisy_labels' in test_df:
            del test_df['noisy_labels']

        preds = list(zip(id_te, test_probs[:, 1]))
        preds_df = pd.DataFrame(preds, columns=['com_id', 'noisy_labels'])
        new_test_df = test_df.merge(preds_df, on='com_id', how='left')
        return new_test_df

    def split_training_data(self, train_df, splits=2):
        train_dfs = np.array_split(train_df, splits)
        return train_dfs
