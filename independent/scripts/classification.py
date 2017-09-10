"""
Module that classifies data using an independent model.
"""
import os
from scipy.sparse import hstack
from scipy.sparse import csr_matrix


class Classification:
    """Class to classify data using an independent model."""

    def __init__(self, config_obj, content_features_obj, graph_features_obj,
                 relational_features_obj, util_obj):
        """initializes object dependencies."""

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

    # private
    def define_file_folders(self):
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
        feats_df = coms_df.merge(r_df, on='com_id', how='left')
        if g_df is not None:
            feats_df = feats_df.merge(g_df, on='user_id', how='left').fillna(0)
        feats_df = feats_df.merge(c_df, on='com_id', how='left')
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

    def transform(self, df, c_m, c_df, g_df, r_df, feats_list):
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

    # public
    def main(self, train_df, val_df, test_df):
        """Constructs paths, merges data, converts, and processes data to be
        read by the independent model.
        train_df: original training comments dataframe.
        test_df: original testing comments dataframe."""
        fold = self.config_obj.fold
        classifier = self.config_obj.classifier
        plot_features = not self.config_obj.ngrams
        featureset = 'all'

        image_f, pred_f, model_f = self.define_file_folders()
        c_tr_m, c_va_m, c_te_m, c_df, c_feats = self.cf_obj.build(train_df,
                val_df, test_df)
        g_df, g_feats = self.gf_obj.build(train_df, val_df, test_df)
        r_df, r_feats = self.rf_obj.build(train_df, val_df, test_df)

        feats = c_feats + g_feats + r_feats
        x_tr, y_tr, _ = self.transform(train_df, c_tr_m, c_df, g_df, r_df,
                feats)
        x_va, y_va, id_va = self.transform(val_df, c_va_m, c_df, g_df, r_df,
                feats)
        x_te, y_te, id_te = self.transform(test_df, c_te_m, c_df, g_df, r_df,
                feats)
        self.util_obj.classify(x_tr, y_tr, x_va, y_va, x_te, y_te,
                id_va, id_te, fold, feats, featureset, image_f, pred_f,
                model_f, classifier=classifier, save_feat_plot=plot_features)
