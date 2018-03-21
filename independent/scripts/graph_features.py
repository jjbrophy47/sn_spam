"""
Module that creates graphical features from a list of follow actions.
"""
import pandas as pd


class GraphFeatures:

    def __init__(self, config_obj, util_obj):
        self.config_obj = config_obj
        self.util_obj = util_obj

    # public
    def build(self, df, fw=None):
        """Specifies user behavior features.
        df: messages dataframe.
        fw: file writer.
        Returns dataframe of comment ids and a list of graph features."""
        self.util_obj.out('obtaining graph features...')
        feats_df, feats_list = self.build_features(df)
        return feats_df, feats_list

    # private
    def build_features(self, df):
        feats_df, feats_list = None, None

        if self.config_obj.domain == 'adclicks':
            feats_df, feats_list = self._adclicks(df)
        elif self.config_obj.domain == 'soundcloud':
            feats_df, feats_list = self.soundcloud(df)
        elif self.config_obj.domain == 'youtube':
            feats_df, feats_list = self.youtube(df)
        elif self.config_obj.domain == 'twitter':
            feats_df, feats_list = self.twitter(df)
        elif self.config_obj.domain == 'ifwe':
            feats_df, feats_list = self.ifwe(df)
        elif self.config_obj.domain == 'toxic':
            feats_df, feats_list = self.toxic(df)
        elif self.config_obj.domain == 'yelp_hotel':
            feats_df, feats_list = self.yelp_hotel(df)
        elif self.config_obj.domain == 'yelp_restaurant':
            feats_df, feats_list = self.yelp_restaurant(df)

        return feats_df, feats_list

    def _adclicks(self, df):
        feats_df = pd.DataFrame(df['com_id'])
        feats_list = []
        return feats_df, feats_list

    def _soundcloud(self, cf):
        featureset = self.config_obj.featureset
        feats_df = pd.DataFrame(cf['com_id'])

        if any(x in featureset for x in ['graph', 'all']):
            feats_list = ['pagerank', 'triangle_count', 'core_id',
                          'out_degree', 'in_degree']
        return feats_df, feats_list

    def youtube(self, cf):
        feats_df = pd.DataFrame(cf['com_id'])
        feats_list = []
        return feats_df, feats_list

    def twitter(self, cf):
        featureset = self.config_obj.featureset
        feats_df = pd.DataFrame(cf['com_id'])

        feats_list = []
        if any(x in featureset for x in ['graph', 'all']):
            feats_list = ['pagerank', 'triangle_count', 'core_id',
                          'out_degree', 'in_degree']
        return feats_df, feats_list

    def toxic(self, cf):
        feats_df = pd.DataFrame(cf['com_id'])
        feats_list = []
        return feats_df, feats_list

    def ifwe(self, cf):
        featureset = self.config_obj.featureset
        feats_df = pd.DataFrame(cf['com_id'])

        if any(x in featureset for x in ['graph', 'all']):
            feats_list = []
            graph_algorithms = ['pagerank', 'triangle_count', 'core_id',
                                'color_id', 'component_id', 'component_size',
                                'out_degree', 'in_degree']
            for x in range(1, 8):
                feats_list.extend([str(x) + '_' + g for g in graph_algorithms])

        return feats_df, feats_list

    def yelp_hotel(self, cf):
        feats_df = pd.DataFrame(cf['com_id'])
        feats_list = []
        return feats_df, feats_list

    def yelp_restaurant(self, cf):
        feats_df = pd.DataFrame(cf['com_id'])
        feats_list = []
        return feats_df, feats_list
