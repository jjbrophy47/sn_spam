"""
Module that creates graphical features from a list of follow actions.
"""
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
        """Specifies user behavior features.
        train_df: training dataframe.
        test_df: testing dataframe.
        Returns dataframe of comment ids and a list of graph features."""
        self.util_obj.start('loading graph features...')
        tr_feats_df, _ = self.build_features(train_df)
        te_feats_df, feats_list = self.build_features(test_df)
        feats_df = pd.concat([tr_feats_df, te_feats_df])
        self.util_obj.end()

        return feats_df, feats_list

    # private
    def build_features(self, cf):
        """Selector to build features for the chosen domain.
        cf: comments dataframe.
        Returns dataframe of comment ids and a list of graph features."""
        feats_df, feats_list = None, None

        if self.config_obj.domain == 'soundcloud':
            feats_df, feats_list = self.soundcloud(cf)
        elif self.config_obj.domain == 'youtube':
            feats_df, feats_list = self.youtube(cf)
        elif self.config_obj.domain == 'twitter':
            feats_df, feats_list = self.twitter(cf)
        elif self.config_obj.domain == 'ifwe':
            feats_df, feats_list = self.ifwe(cf)
        elif self.config_obj.domain == 'yelp_hotel':
            feats_df, feats_list = self.yelp_hotel(cf)
        elif self.config_obj.domain == 'yelp_restaurant':
            feats_df, feats_list = self.yelp_restaurant(cf)

        return feats_df, feats_list

    def soundcloud(self, cf):
        """Specifies which graph features to keep.
        cf: comments dataframe.
        Returns dataframe with comment ids and a list of graph feaures."""
        feats_df = pd.DataFrame(cf['com_id'])
        feats_list = ['pagerank', 'triangle_count', 'core_id', 'out_degree',
                'in_degree']
        return feats_df, feats_list

    def youtube(self, cf):
        """Currently no graph features for youtube users.
        cf: comments dataframe.
        Returns dataframe with comment ids and an empty list."""
        feats_df = pd.DataFrame(cf['com_id'])
        feats_list = []
        return feats_df, feats_list

    def twitter(self, cf):
        """Specifies which graph features to keep.
        cf: comments dataframe.
        Returns dataframe with comment ids and a list of graph feaures."""
        feats_df = pd.DataFrame(cf['com_id'])
        feats_list = ['pagerank', 'triangle_count', 'core_id', 'out_degree',
                'in_degree']
        return feats_df, feats_list

    def ifwe(self, cf):
        """Specifies which graph features to keep.
        cf: comments dataframe.
        Returns dataframe with comment ids and a list of graph feaures."""
        feats_df = pd.DataFrame(cf['com_id'])
        feats_list = []
        graph_algorithms = ['pagerank', 'triangle_count', 'core_id',
                'color_id', 'component_id', 'component_size', 'out_degree',
                'in_degree']

        for x in range(1, 8):
            feats_list.extend([str(x) + '_' + g for g in graph_algorithms])
        return feats_df, feats_list

    def yelp_hotel(self, cf):
        """Currently no graph features for yelp_hotel reviewers.
        cf: comments dataframe.
        Returns dataframe with comment ids and an empty list."""
        feats_df = pd.DataFrame(cf['com_id'])
        feats_list = []
        return feats_df, feats_list

    def yelp_restaurant(self, cf):
        """Currently no graph features for yelp_restaurant reviewers.
        cf: comments dataframe.
        Returns dataframe with comment ids and an empty list."""
        feats_df = pd.DataFrame(cf['com_id'])
        feats_list = []
        return feats_df, feats_list
