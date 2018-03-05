"""
Module that classifies data using an independent model.
"""
import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.sparse import csr_matrix


class Classification:
    """Class to classify data using an independent model."""

    def __init__(self, config_obj, content_features_obj, graph_features_obj,
                 relational_features_obj, util_obj):
        """Initializes object dependencies."""

        self.config_obj = config_obj
        """User settings object."""
        self.cf_obj = content_features_obj
        """Class to construct features based on text content."""
        self.gf_obj = graph_features_obj
        """Class to construct graphical features."""
        self.rf_obj = relational_features_obj
        """Class to construct relational features."""
        self.util_obj = util_obj
        """Utility class that does graphing, classification, etc."""

    # public
    def main(self, train_df, test_df, dset='test', fw=None, stacking=0):
        """Constructs paths, merges data, converts, and processes data to be
        read by the independent model.
        train_df: original training comments dataframe.
        test_df: original testing comments dataframe.
        dset: datatset to test (e.g. 'val', 'test')."""
        if stacking > 0:
            self.do_stacking(train_df, test_df, dset, fw, stacking=stacking)
        else:
            self.do_normal(train_df, test_df, dset, fw)

    # private
    def do_stacking(self, train_df, test_df, dset='test', stacking=1, fw=None):
        fold = self.config_obj.fold
        classifier = self.config_obj.classifier

        image_f, pred_f, model_f = self.file_folders()
        train_dfs = self.split_training_data(train_df, splits=stacking + 1)
        test_df = test_df.copy()

        for i in range(stacking):
            data_te = self.build_and_merge(train_dfs[i], test_df, 'te', fw=fw)
            learner = self.util_obj.train(data_te, classifier=classifier,
                                          fw=fw)
            te_preds, ids = self.util_obj.test(data_te, learner, fw=fw)
            test_df = self.append_preds(test_df, te_preds, ids)

            for j in range(i + 1, len(train_dfs)):
                data_tr = self.build_and_merge(train_dfs[i], train_dfs[j],
                                               'tr', fw=fw)
                tr_preds, ids = self.util_obj.test(data_tr, learner, fw=fw)
                train_dfs[j] = self.append_preds(train_dfs[j], tr_preds, ids)

        data_te = self.build_and_merge(train_dfs[len(stacking)], test_df,
                                       'te', fw=fw)
        learner = self.util_obj.train(data_te, classifier=classifier, fw=fw)
        te_preds, ids = self.util_obj.test(data_te, learner, fw=fw)
        self.util_obj.evaluate(data_te, te_preds, fw=fw)
        self.util_obj.save_preds(te_preds, ids, fold, pred_f, dset)

    # # private
    # def do_stacking(self, train_df, test_df, dset='test', fw=None):
    #     fold = self.config_obj.fold
    #     classifier = self.config_obj.classifier
    #
    #     image_f, pred_f, model_f = self.file_folders()
    #     train1_df, train2_df = self.split_training_data(train_df)
    #
    #     # train base learner using train1 set.
    #     data = self.build_and_merge(train1_df, train2_df, 'train1', fw=fw)
    #     base_learner = self.util_obj.train(data, classifier=classifier, fw=fw)
    #
    #     # add predictions to train2 and train stacked model using train2 set.
    #     train2_probs, id_tr2 = self.util_obj.test(data, base_learner, fw=fw)
    #     new_train2_df = self.append_preds(train2_df, train2_probs, id_tr2)
    #     data = self.build_and_merge(new_train2_df, test_df, 'train2', fw=fw)
    #     real_learner = self.util_obj.train(data, classifier=classifier, fw=fw)
    #
    #     # get predictions for test set using the base learner.
    #     data = self.build_and_merge(train_df, test_df, dset, fw=fw)
    #     base_learner2 = self.util_obj.train(data, classifier=classifier, fw=fw)
    #     test_probs, id_te = self.util_obj.test(data, base_learner2, fw=fw)
    #
    #     # add predictions to test set and test stacked model on the test set.
    #     new_test_df = self.append_preds(test_df, test_probs, id_te)
    #     data = self.build_and_merge(new_train2_df, new_test_df, dset, fw=fw)
    #     test_probs, id_te = self.util_obj.test(data, real_learner, fw=fw)
    #     self.util_obj.evaluate(data, test_probs, fw=fw)
    #     self.util_obj.save_preds(test_probs, id_te, fold, pred_f, dset)

    def do_normal(self, train_df, test_df, dset='test', fw=None):
        fold = self.config_obj.fold
        classifier = self.config_obj.classifier

        image_f, pred_f, model_f = self.file_folders()
        train1_df, train2_df = self.split_training_data(train_df)

        # train base learner using training set.
        data = self.build_and_merge(train_df, test_df, dset, fw=fw)
        base_learner = self.util_obj.train(data, classifier=classifier, fw=fw)

        # get predictions for test set using the base learner.
        test_probs, id_te = self.util_obj.test(data, base_learner, fw=fw)
        self.util_obj.evaluate(data, test_probs, fw=fw)
        self.util_obj.save_preds(test_probs, id_te, fold, pred_f, dset)

    def file_folders(self):
        """Returns absolute path directories."""
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
        """Merges the features dataframes with the comments dataframe.
        coms_df: original data.
        c_df: content features.
        g_df: graph features.
        r_df: relational features.
        Returns the merged dataframe."""
        feats_df = coms_df.merge(c_df, on='com_id', how='left')
        feats_df = feats_df.merge(g_df, on='com_id', how='left')
        feats_df = feats_df.merge(r_df, on='com_id', how='left')
        return feats_df

    def drop_columns(self, feats_df, feats_list):
        """Drops columns not related to the features (e.g. com_id, etc.).
        feats_df: merged feature dataframe.
        feats_list: list of features to be used for classification.
        Returns the filtered dataframe."""
        return feats_df[feats_list]

    def dataframe_to_matrix(self, feats_df):
        """Converts the dataframe to a matrix.
        feats_df: filtered dataframe.
        Returns the dataframe as a matrix."""
        return csr_matrix(feats_df.as_matrix())

    def stack_matrices(self, feats_m, c_csr):
        """Merges the matrices into one matrix.
        feats_m: coverted feature matrix.
        c_csr: sparse ngram matrix.
        Returns sparse merged matrix."""
        stack = [feats_m]
        if c_csr is not None:
            stack.append(c_csr)
        return hstack(stack).tocsr()

    def extract_labels(self, coms_df):
        """Extracts the ground truth labels.
        coms_df: original data.
        Returns a tuple of numpy arrays: labels and ids."""
        return coms_df['label'].values, coms_df['com_id'].values

    def prepare(self, df, c_m, c_df, g_df, r_df, feats_list):
        """Takes specified dataset, ngram sparse matrix, and feature frames
        and processes them into features and labels to feed into the model.
        df: training, validation, or testing dataframe.
        c_m: ngram sparse matrix.
        c_df: content features dataframe.
        g_df: graph features dataframe.
        r_df: relational features dataframe.
        feats_list: list of feature names.
        Returns features x as a sparse matrix, and labels y as an ndarray."""
        feats_df = self.merge(df, c_df, g_df, r_df)
        feats_df = self.drop_columns(feats_df, feats_list)
        feats_m = self.dataframe_to_matrix(feats_df)
        x = self.stack_matrices(feats_m, c_m)
        y, ids = self.extract_labels(df)
        return x, y, ids

    def build_and_merge(self, train_df, test_df, dset, fw=None):
        """Builds the featuresets and merges the features into one dataframe.
        train_df: training set dataframe.
        test_df: test set dataframe.
        dset: dataset to test (e.g. 'val', 'test').
        Returns the training and testing feature and label dataframes, and a
                list of all computed features."""
        m_tr, m_te, c_df, c_feats = self.cf_obj.build(train_df, test_df, dset,
                                                      fw=fw)
        g_df, g_feats = self.gf_obj.build(train_df, test_df, fw=fw)
        r_df, r_feats = self.rf_obj.build(train_df, test_df, dset, fw=fw)

        self.util_obj.start('merging features...', fw=fw)
        feats = c_feats + g_feats + r_feats
        x_tr, y_tr, _ = self.prepare(train_df, m_tr, c_df, g_df, r_df,
                                     feats)
        x_te, y_te, id_te = self.prepare(test_df, m_te, c_df, g_df, r_df,
                                         feats)
        self.util_obj.end(fw=fw)
        return x_tr, y_tr, x_te, y_te, id_te, feats

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
