"""
Module that creates features based on the text of the comments.
"""
import numpy as np
import pandas as pd
import scipy.sparse as ss
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer


class ContentFeatures:

    def __init__(self, config_obj, util_obj):
        self.config_obj = config_obj
        self.util_obj = util_obj

    # public
    def build(self, df, dset, cv=None):
        """Builds content features based on the text in the data.
        df: comments dataframe.
        dset: dataset (e.g. 'val', 'test').
        Returns ngram matrices for each dataset, content features dataframe,
                and a list of features created."""
        feats_df, feats = self._build_features(df)
        m, cv = self._ngrams(df, cv=cv)
        return m, feats_df, feats, cv

    # private
    def _ngrams(self, df, cv=None):
        domain = self.config_obj.domain
        featuresets = self.config_obj.featuresets

        m = None

        if domain not in ['ifwe', 'adclicks'] and \
                any(x in featuresets for x in ['ngrams', 'all']):
            m, cv = self._build_ngrams(df, cv=cv)
        return m, cv

    def _count_vectorizer(self):
        cv = CountVectorizer(stop_words='english', min_df=1,
                             ngram_range=(3, 3), max_df=1.0,
                             max_features=10000, analyzer='char_wb',
                             binary=True, vocabulary=None, dtype=np.int32)
        return cv

    def _build_ngrams(self, df, cv=None):
        t1 = self.util_obj.out('building ngrams...')
        str_list = df[:]['text'].tolist()

        if cv is None:
            cv = self._count_vectorizer()
            ngrams_m = cv.fit_transform(str_list)
        else:
            ngrams_m = cv.transform(str_list)

        id_m = ss.lil_matrix((len(df), 1))
        ngrams_csr = ss.hstack([id_m, ngrams_m]).tocsr()
        self.util_obj.time(t1)
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
        elif self.config_obj.domain == 'russia':
            features_df, features_list = self._russia(df)
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
        featuresets = self.config_obj.featuresets
        feats_df = pd.DataFrame(df['com_id'])

        base = ['ip', 'app', 'device', 'os', 'channel']
        usr = ['ip', 'device', 'os']
        app = ['app']
        usr_app = usr + app
        mfh = [4, 5, 9, 10, 13, 14]  # most freq hours in test
        lfh = [6, 11, 15]  # least freq hours in test
        in_h = lambda x: 1 if x in mfh else 2 if x in lfh else 3
        n_app = ['app', 'wday', 'in_h']
        n_ip = ['ip', 'wday', 'in_h']
        n_ip_app = ['ip', 'wday', 'in_h', 'app']
        n_ip_os = ['ip', 'wday', 'in_h', 'os']
        n_ip_app_os = ['ip', 'wday', 'in_h', 'app', 'os']

        if any(x in featuresets for x in ['content', 'all']):
            t1 = self.util_obj.out('building content features...')
            df['click_time'] = pd.to_datetime(df['click_time'])
            feats_df[base] = df[base]
            feats_df['wday'] = df['click_time'].dt.dayofweek
            feats_df['hour'] = df['click_time'].dt.hour
            feats_df['min'] = df['click_time'].dt.minute
            feats_df['usr_newness'] = df.groupby(usr).cumcount()
            feats_df['usr_count'] = self._count(usr, df)
            feats_df['usr_app_newness'] = df.groupby(usr_app).cumcount()
            feats_df['usr_app_count'] = self._count(usr_app, df)
            feats_df['in_h'] = feats_df['hour'].apply(in_h)
            feats_df['n_app'] = self._count(n_app, feats_df)
            feats_df['n_ip'] = self._count(n_ip, feats_df)
            feats_df['n_ip_app'] = self._count(n_ip_app, feats_df)
            feats_df['n_ip_os'] = self._count(n_ip_os, feats_df)
            feats_df['n_ip_app_os'] = self._count(n_ip_app_os, feats_df)
            self.util_obj.time(t1)

        features_list = list(feats_df)

        features_list.remove('com_id')
        features_list.remove('ip')
        features_list.remove('in_h')
        features_list.remove('wday')
        features_list.remove('hour')
        feats_df = feats_df.drop(base + ['in_h', 'wday', 'hour'], axis=1)
        return feats_df, features_list

    def _soundcloud(self, df):
        featuresets = self.config_obj.featuresets
        feats_df = pd.DataFrame(df['com_id'])
        feats_list = []

        if any(x in featuresets for x in ['content', 'all']):
            t1 = self.util_obj.out('building content features...')
            feats_df['com_num_chars'] = df['text'].str.len()
            feats_df['com_has_link'] = df['text'].str.contains('http')
            feats_df['com_has_link'] = feats_df['com_has_link'].astype(int)
            feats_list.extend(list(feats_df))
            feats_list.remove('com_id')
            feats_list.append('polarity')
            feats_list.append('subjectivity')
            self.util_obj.time(t1)

        return feats_df, feats_list

    def _youtube(self, df):
        featuresets = self.config_obj.featuresets
        feats_df = pd.DataFrame(df['com_id'])
        feats_list = []

        if any(x in featuresets for x in ['content', 'all']):
            t1 = self.util_obj.out('building content features...')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            feats_df['com_num_chars'] = df['text'].str.len()
            feats_df['com_weekday'] = df['timestamp'].dt.dayofweek
            feats_df['com_hour'] = df['timestamp'].dt.hour
            feats_list.extend(list(feats_df))
            feats_list.remove('com_id')
            feats_list.append('polarity')
            feats_list.append('subjectivity')
            self.util_obj.time(t1)

        return feats_df, feats_list

    def _twitter(self, df):
        featuresets = self.config_obj.featuresets
        feats_df = pd.DataFrame(df['com_id'])
        feats_list = []

        if any(x in featuresets for x in ['content', 'all']):
            t1 = self.util_obj.out('building content features...')
            feats_df['com_num_chars'] = df['text'].str.len()
            feats_df['com_num_hashtags'] = df['text'].str.count('#')
            feats_df['com_num_mentions'] = df['text'].str.count('@')
            feats_df['com_num_links'] = df['text'].str.count('http')
            feats_df['com_num_retweets'] = df['text'].str.count('RT')
            feats_list.extend(list(feats_df))
            feats_list.remove('com_id')
            feats_list.append('polarity')
            feats_list.append('subjectivity')
            self.util_obj.time(t1)

        return feats_df, feats_list

    def _russia(self, df):
        featuresets = self.config_obj.featuresets
        feats_df = pd.DataFrame(df['com_id'])

        if any(x in featuresets for x in ['content', 'all']):
            t1 = self.util_obj.out('building content features...')
            polarity = lambda x: TextBlob(x).sentiment.polarity
            subjectivity = lambda x: TextBlob(x).sentiment.subjectivity
            feats_df['com_num_chars'] = df['text'].str.len()
            feats_df['com_num_hashtags'] = df['text'].str.count('#')
            feats_df['com_num_mentions'] = df['text'].str.count('@')
            feats_df['com_num_links'] = df['text'].str.count('http')
            feats_df['com_num_retweets'] = df['text'].str.count('RT')
            feats_df['com_polarity'] = df['text'].apply(polarity)
            feats_df['com_subjectivity'] = df['text'].apply(subjectivity)
            self.util_obj.time(t1)

        features_list = list(feats_df)

        if any(x in featuresets for x in ['base', 'all']):
            self.util_obj.out('building base features...')
            features_list.extend(['user_favourites_count',
                                  'user_followers_count', 'user_friends_count',
                                  'user_statuses_count', 'user_verified'])

        features_list.remove('com_id')
        return feats_df, features_list

    def _toxic(self, df):
        featuresets = self.config_obj.featuresets
        feats_df = pd.DataFrame(df['com_id'])
        feats_list = []

        if any(x in featuresets for x in ['content', 'all']):
            t1 = self.util_obj.out('building content features...')
            feats_df['com_num_chars'] = df['text'].str.len()
            feats_df['com_num_links'] = df['text'].str.count('http')
            feats_list.extend(list(feats_df))
            feats_list.remove('com_id')
            feats_list.append('polarity')
            feats_list.append('subjectivity')
            self.util_obj.time(t1)

        return feats_df, feats_list

    def _ifwe(self, df):
        featuresets = self.config_obj.featuresets
        features_df = pd.DataFrame(df['com_id'])

        features_list = list(features_df)

        if any(x in featuresets for x in ['base', 'all']):
            self.util_obj.out('building base features...')
            features_list.extend(['sex_id', 'time_passed_id', 'age_id'])

        features_list.remove('com_id')
        return features_df, features_list

    def _yelp_hotel(self, df):
        featuresets = self.config_obj.featuresets
        features_df = pd.DataFrame(df['com_id'])

        if any(x in featuresets for x in ['content', 'all']):
            self.util_obj.out('building content features...')
            features_df['com_num_chars'] = df['text'].str.len()
            features_df['com_num_links'] = df['text'].str.count('http')

        features_list = list(features_df)
        features_list.remove('com_id')
        return features_df, features_list

    def _yelp_restaurant(self, df):
        featuresets = self.config_obj.featuresets
        features_df = pd.DataFrame(df['com_id'])

        if any(x in featuresets for x in ['content', 'all']):
            self.util_obj.out('building content features...')
            features_df['com_num_chars'] = df['text'].str.len()
            features_df['com_num_links'] = df['text'].str.count('http')

        features_list = list(features_df)
        features_list.remove('com_id')
        return features_df, features_list

    def _count(self, cols, df):
        qf1 = df.groupby(cols).size().reset_index().rename(columns={0: 'size'})
        qf2 = df.merge(qf1, how='left')
        return list(qf2['size'])
