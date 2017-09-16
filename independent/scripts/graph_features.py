"""
Module that creates graphical features from a list of follow actions.
"""
import os
import pandas as pd


class GraphFeatures:
    """Class that handles all operations for creating graphical features."""

    def __init__(self, config_obj, util_obj):
        """Initialize object dependencies."""

        self.config_obj = config_obj
        """User setttings."""
        self.util_obj = util_obj
        """General utility methods."""

    # public
    def build(self, train_df, test_df):
        """Builds or loads graphical features.
        train_df: training dataframe.
        test_df: testing dataframe.
        Returns graph features dataframe and list."""
        domain = self.config_obj.domain
        feats_df, feature_list = None, []

        if domain == 'soundcloud' or domain == 'twitter':
            data_f, gl_f, feat_f = self.define_file_folders()

            if self.util_obj.check_file(feat_f + 'graph_features.csv'):
                self.util_obj.start('loading graph features...')
                feats_df = pd.read_csv(feat_f + 'graph_features.csv')
                feature_list = feats_df.columns.tolist()
                feature_list.remove('user_id')
                self.util_obj.end()
        return feats_df, feature_list

    # private
    def define_file_folders(self):
        """Returns absolute path directories."""
        ind_dir = self.config_obj.ind_dir
        domain = self.config_obj.domain

        data_f = ind_dir + 'data/' + domain + '/'
        gl_f = ind_dir + 'data/' + domain + '/graphlab/'
        feat_f = ind_dir + 'output/' + domain + '/features/'
        if not os.path.exists(feat_f):
            os.makedirs(feat_f)
        if not os.path.exists(gl_f):
            os.makedirs(gl_f)
        return data_f, gl_f, feat_f
