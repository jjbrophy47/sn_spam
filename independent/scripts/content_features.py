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
        self.util_obj.out('building content features...')
        feats_df, feats = self._build_features(df)
        m, cv = self._ngrams(df, cv=cv, fw=fw)
        return m, feats_df, feats, cv

    # private
    def _ngrams(self, df, cv=None, fw=None):
        use_ngrams = self.config_obj.ngrams
        domain = self.config_obj.domain
        featureset = self.config_obj.featureset

        m = None

        if use_ngrams and domain not in ['ifwe', 'adclicks'] and \
                any(x in featureset for x in ['content', 'all']):
            m, cv = self._build_ngrams(df, cv=cv, fw=fw)
        return m, cv

    def _count_vectorizer(self):
        cv = CountVectorizer(stop_words='english', min_df=1,
                             ngram_range=(3, 3), max_df=1.0,
                             max_features=10000, analyzer='char_wb',
                             binary=True, vocabulary=None, dtype=np.int32)
        return cv

    def _build_ngrams(self, df, cv=None, fw=None):
        self.util_obj.write('constructing ngrams...', fw=fw)
        str_list = df[:]['text'].tolist()

        if cv is None:
            cv = self._count_vectorizer()
            ngrams_m = cv.fit_transform(str_list)
        else:
            ngrams_m = cv.transform(str_list)

        id_m = ss.lil_matrix((len(df), 1))
        ngrams_csr = ss.hstack([id_m, ngrams_m]).tocsr()
        return ngrams_csr, cv

    def _build_features(self, df):
        features_df, features_list = None, None

        if self.config_obj.domain not in ['ifwe', 'adclicks']:
            df['text'] = df['text'].fillna('')

        if self.config_obj.domain == 'adclicks':
            features_df, features_list = self._adclicks(df)
        if self.config_obj.domain == 'soundcloud':
            features_df, features_list = self._soundcloud(df)
        elif self.config_obj.domain == 'youtube':
            features_df, features_list = self._youtube(df)
        elif self.config_obj.domain == 'twitter':
            features_df, features_list = self._twitter(df)
        elif self.config_obj.domain == 'toxic':
            features_df, features_list = self._toxic(df)
        elif self.config_obj.domain == 'ifwe':
            features_df, features_list = self._ifwe(df)
        elif self.config_obj.domain == 'yelp_hotel':
            features_df, features_list = self._yelp_hotel(df)
        elif self.config_obj.domain == 'yelp_restaurant':
            features_df, features_list = self._yelp_restaurant(df)

        return features_df, features_list

    def _adclicks(self, df):
        featureset = self.config_obj.featureset
        features_df = pd.DataFrame(df['com_id'])

        if any(x in featureset for x in ['content', 'all']):
            df['click_time'] = pd.to_datetime(df['click_time'])
            features_df['com_weekday'] = df['click_time'].dt.dayofweek
            features_df['com_hour'] = df['click_time'].dt.hour
            features_df['com_min'] = df['click_time'].dt.minute

        features_list = list(features_df)

        if any(x in featureset for x in ['base', 'all']):
            features_list.extend(['ip', 'app', 'device', 'os', 'channel'])

        features_list.remove('com_id')
        return features_df, features_list

    def _soundcloud(self, df):
        features_df = pd.DataFrame(df['com_id'])
        features_df['com_num_chars'] = df['text'].str.len()
        features_df['com_has_link'] = df['text'].str.contains('http')
        features_df['com_has_link'] = features_df['com_has_link'].astype(int)
        features_list = list(features_df)
        features_list.remove('com_id')
        return features_df, features_list

    def _youtube(self, df):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        features_df = pd.DataFrame(df['com_id'])
        features_df['com_num_chars'] = df['text'].str.len()
        features_df['com_weekday'] = df['timestamp'].dt.dayofweek
        features_df['com_hour'] = df['timestamp'].dt.hour
        features_list = list(features_df)
        features_list.remove('com_id')
        return features_df, features_list

    def _twitter(self, df):
        features_df = pd.DataFrame(df['com_id'])
        features_df['com_num_chars'] = df['text'].str.len()
        features_df['com_num_hashtags'] = df['text'].str.count('#')
        features_df['com_num_mentions'] = df['text'].str.count('@')
        features_df['com_num_links'] = df['text'].str.count('http')
        features_df['com_num_retweets'] = df['text'].str.count('RT')
        features_list = list(features_df)
        features_list.remove('com_id')
        return features_df, features_list

    def _toxic(self, df):
        feats_df = pd.DataFrame(df['com_id'])
        feats_df['com_num_chars'] = df['text'].str.len()
        feats_df['com_num_links'] = df['text'].str.count('http')
        features_list = list(feats_df)
        features_list.remove('com_id')
        return feats_df, features_list

    def _ifwe(self, df):
        features_df = pd.DataFrame(df['com_id'])
        features_list = ['sex_id', 'time_passed_id', 'age_id']
        return features_df, features_list

    def _yelp_hotel(self, df):
        features_df = pd.DataFrame(df['com_id'])
        features_df['com_num_chars'] = df['text'].str.len()
        features_df['com_num_links'] = df['text'].str.count('http')
        features_list = list(features_df)
        features_list.remove('com_id')
        return features_df, features_list

    def _yelp_restaurant(self, df):
        features_df = pd.DataFrame(df['com_id'])
        features_df['com_num_chars'] = df['text'].str.len()
        features_df['com_num_links'] = df['text'].str.count('http')
        features_list = list(features_df)
        features_list.remove('com_id')
        return features_df, features_list
