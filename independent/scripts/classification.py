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
    def main(self, train_df, test_df, dset='test', fw=None):
        """Constructs paths, merges data, converts, and processes data to be
        read by the independent model.
        train_df: original training comments dataframe.
        test_df: original testing comments dataframe.
        dset: datatset to test (e.g. 'val', 'test')."""
        stacking = self.config_obj.stacking

        if stacking > 0:
            self.do_stacking(train_df, test_df, dset, stacking=stacking, fw=fw)
        else:
            self.do_normal(train_df, test_df, dset, fw)

    # private
    def do_stacking(self, train_df, test_df, dset='test', stacking=1, fw=None):
        print('doing stacking with %d stack(s)...' % stacking)
        fold = self.config_obj.fold
        classifier = self.config_obj.classifier

        image_f, pred_f, model_f = self.file_folders()
        trains = self.split_training_data(train_df, splits=stacking + 1)
        test_df = test_df.copy()

        for i in range(len(trains)):
            d_tr, cv = self.build_and_merge(trains[i], 'train', fw=fw)
            learner = self.util_obj.train(d_tr, clf=classifier, fw=fw)

            for j in range(i + 1, len(trains)):
                d_te, _ = self.build_and_merge(trains[j], 'test', cv=cv, fw=fw)
                te_preds, ids = self.util_obj.test(d_te, learner, fw=fw)
                trains[j] = self.append_preds(trains[j], te_preds, ids)

            d_te, _ = self.build_and_merge(test_df, 'test', cv=cv, fw=fw)
            te_preds, ids = self.util_obj.test(d_te, learner, fw=fw)
            test_df = self.append_preds(test_df, te_preds, ids)

        self.util_obj.evaluate(d_te, te_preds, fw=fw)
        self.util_obj.save_preds(te_preds, ids, fold, pred_f, dset)

        if not self.config_obj.ngrams:
            _, _, _, feats = d_te
            self.util_obj.plot_features(learner, 'lr', feats, image_f + 'a')

    def do_normal(self, train_df, test_df, dset='test', fw=None):
        fold = self.config_obj.fold
        classifier = self.config_obj.classifier

        image_f, pred_f, model_f = self.file_folders()

        # train base learner using training set.
        d_tr, cv = self.build_and_merge(train_df, 'train', fw=fw)
        learner = self.util_obj.train(d_tr, clf=classifier, fw=fw)

        # test learner on test set.
        d_te, _ = self.build_and_merge(test_df, 'test', fw=fw)
        y_score, ids = self.util_obj.test(d_te, learner, fw=fw)
        self.util_obj.evaluate(d_te, y_score, fw=fw)
        self.util_obj.save_preds(y_score, ids, fold, pred_f, dset)

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

    def extract_labels(self, coms_df):
        return coms_df['label'].values, coms_df['com_id'].values

    def prepare(self, df, c_m, c_df, g_df, r_df, feats_list):
        feats_df = self.merge(df, c_df, g_df, r_df)
        feats_df = self.drop_columns(feats_df, feats_list)
        feats_m = self.dataframe_to_matrix(feats_df)
        x = self.stack_matrices(feats_m, c_m)
        y, ids = self.extract_labels(df)
        return x, y, ids

    def build_and_merge(self, df, dset, cv=None, fw=None):
        m, c_df, c_feats, cv = self.cf_obj.build(df, dset, fw=fw)
        g_df, g_feats = self.gf_obj.build(df, fw=fw)
        r_df, r_feats = self.rf_obj.build(df, dset, fw=fw)

        self.util_obj.start('merging features...', fw=fw)
        feats = c_feats + g_feats + r_feats
        x, y, ids = self.prepare(df, m, c_df, g_df, r_df, feats)
        print(x.shape)
        self.util_obj.end(fw=fw)
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
