"""
Module that creates features based on the text of the comments.
"""
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.feature_extraction.text import CountVectorizer


class ContentFeatures:

    def __init__(self, config_obj, util_obj):
        self.config_obj = config_obj
        self.util_obj = util_obj

    # public
    def build(self, df, dset, cv=None, fw=None):
        """Builds content features based on the text in the data.
        df: comments dataframe.
        dset: dataset (e.g. 'val', 'test').
        Returns ngram matrices for each dataset, content features dataframe,
                and a list of features created."""
        self.util_obj.start('building content features...', fw=fw)
        feats_df, feats = self.build_features(df)
        m, cv = self.ngrams(df, cv=cv, fw=fw)
        self.util_obj.end(fw=fw)

        return m, feats_df, feats, cv

    # private
    def ngrams(self, df, cv=None, fw=None):
        m = None

        if self.config_obj.ngrams and self.config_obj.domain != 'ifwe':
            m, cv = self.build_ngrams(df, cv=cv, fw=fw)
        return m, cv

    def count_vectorizer(self):
        cv = CountVectorizer(stop_words='english', min_df=1,
                             ngram_range=(3, 3), max_df=1.0,
                             max_features=10000, analyzer='char_wb',
                             binary=True, vocabulary=None, dtype=np.int32)
        return cv

    def build_ngrams(self, df, cv=None, fw=None):
        self.util_obj.write('constructing ngrams...', fw=fw)
        str_list = df[:]['text'].tolist()

        if cv is None:
            cv = self.count_vectorizer()
            ngrams_m = cv.fit_transform(str_list)
        else:
            ngrams_m = cv.transform(str_list)

        id_m = ss.lil_matrix((len(df), 1))
        ngrams_csr = ss.hstack([id_m, ngrams_m]).tocsr()
        return ngrams_csr, cv

    def build_features(self, df):
        features_df, features_list = None, None

        if self.config_obj.domain != 'ifwe':
            df['text'] = df['text'].fillna('')

        if self.config_obj.domain == 'soundcloud':
            features_df, features_list = self.soundcloud(df)
        elif self.config_obj.domain == 'youtube':
            features_df, features_list = self.youtube(df)
        elif self.config_obj.domain == 'twitter':
            features_df, features_list = self.twitter(df)
        elif self.config_obj.domain == 'toxic':
            features_df, features_list = self.toxic(df)
        elif self.config_obj.domain == 'ifwe':
            features_df, features_list = self.ifwe(df)
        elif self.config_obj.domain == 'yelp_hotel':
            features_df, features_list = self.yelp_hotel(df)
        elif self.config_obj.domain == 'yelp_restaurant':
            features_df, features_list = self.yelp_restaurant(df)

        return features_df, features_list

    def soundcloud(self, cf):
        """Builds features specifically for soundcloud data.
        cf: comments dataframe.
        Returns features dataframe and list."""
        features_df = pd.DataFrame(cf['com_id'])
        features_df['com_num_chars'] = cf['text'].str.len()
        features_df['com_has_link'] = cf['text'].str.contains('http')
        features_df['com_has_link'] = features_df['com_has_link'].astype(int)
        features_list = list(features_df)
        features_list.remove('com_id')
        return features_df, features_list

    def youtube(self, cf):
        """Builds features specifically for youtube data.
        cf: comments dataframe.
        Returns features dataframe and list."""
        cf['hour'] = cf['timestamp'].astype(str)
        cf['timestamp'] = pd.to_datetime(cf['timestamp'])
        features_df = pd.DataFrame(cf['com_id'])
        features_df['com_num_chars'] = cf['text'].str.len()
        features_df['com_weekday'] = cf['timestamp'].dt.weekday
        features_df['com_hour'] = cf['hour'].str[11:13].astype(int)
        features_list = list(features_df)
        features_list.remove('com_id')
        return features_df, features_list

    def twitter(self, cf):
        """Builds features specifically for twitter data.
        cf: comments dataframe.
        Returns features dataframe and list."""
        features_df = pd.DataFrame(cf['com_id'])
        features_df['com_num_chars'] = cf['text'].str.len()
        features_df['com_num_hashtags'] = cf['text'].str.count('#')
        features_df['com_num_mentions'] = cf['text'].str.count('@')
        features_df['com_num_links'] = cf['text'].str.count('http')
        features_df['com_num_retweets'] = cf['text'].str.count('RT')
        features_list = list(features_df)
        features_list.remove('com_id')
        return features_df, features_list

    def toxic(self, cf):
        feats_df = pd.DataFrame(cf['com_id'])
        feats_df['com_num_chars'] = cf['text'].str.len()
        feats_df['com_num_links'] = cf['text'].str.count('http')
        features_list = list(feats_df)
        features_list.remove('com_id')
        return feats_df, features_list

    def ifwe(self, cf):
        """Specifies demographic features to use.
        cf: comments dataframe.
        Returns dataframe of comment ids and a list of features."""
        features_df = pd.DataFrame(cf['com_id'])
        features_list = ['sex_id', 'time_passed_id', 'age_id']
        return features_df, features_list

    def yelp_hotel(self, cf):
        """Builds features specifically for the yelp_hotel data.
        cf: comments dataframe.
        Returns features dataframe and list."""
        features_df = pd.DataFrame(cf['com_id'])
        features_df['com_num_chars'] = cf['text'].str.len()
        features_df['com_num_links'] = cf['text'].str.count('http')
        features_list = list(features_df)
        features_list.remove('com_id')
        return features_df, features_list

    def yelp_restaurant(self, cf):
        """Builds features specifically for the yelp_restaurant data.
        cf: comments dataframe.
        Returns features dataframe and list."""
        features_df = pd.DataFrame(cf['com_id'])
        features_df['com_num_chars'] = cf['text'].str.len()
        features_df['com_num_links'] = cf['text'].str.count('http')
        features_list = list(features_df)
        features_list.remove('com_id')
        return features_df, features_list
