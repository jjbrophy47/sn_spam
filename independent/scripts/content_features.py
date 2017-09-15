"""
Module that creates features based on the text of the comments.
"""
import pandas as pd
import numpy as np
import scipy.sparse as ss
from sklearn.feature_extraction.text import CountVectorizer


class ContentFeatures:
    """Class that handles all operations to create content features for a
    given domain."""

    def __init__(self, config_obj, util_obj):
        """Initialize object dependencies."""

        self.config_obj = config_obj
        """User settings."""
        self.util_obj = util_obj
        """General utility methods."""

    def settings(self):
        """Returns ngram settings. Strictly term frequency. Selects the
        top 10K features that appear most frequently. Features are then
        converted to binary for each document."""
        ngram_settings = {'stop_words': 'english', 'ngram_range': (3, 3),
                          'max_features': 10000, 'analyzer': 'char_wb',
                          'min_df': 1, 'max_df': 1.0, 'binary': True,
                          'vocabulary': None, 'dtype': np.int32}
        return ngram_settings

    def concat_coms(self, train_df, test_df):
        """Appends the validation and test dataframes onto the trianing set.
        train_df: training set dataframe.
        test_df: testing set dataframe.
        Returns The concatenated dataframe."""
        coms_df = pd.concat([train_df, test_df])
        coms_df['text'] = coms_df['text'].fillna('')
        coms_df = coms_df.reset_index()
        coms_df = coms_df.drop(['index'], axis=1)
        return coms_df

    def count_vectorizer(self, s):
        """Builds a CountVectorizer object with the specified settings.
        s: dict containing ngram settings.
        Returns the initialized CountVectorizer object."""
        cv = CountVectorizer(stop_words=s['stop_words'], min_df=s['min_df'],
                             ngram_range=s['ngram_range'], max_df=s['max_df'],
                             max_features=s['max_features'],
                             analyzer=s['analyzer'], binary=s['binary'],
                             vocabulary=s['vocabulary'], dtype=s['dtype'])
        return cv

    def ngrams(self, cf, s):
        """Constructs ngrams based on the text in the comments.
        cf: comments dataframe.
        s: settings used for ngram construction.
        Returns a compressed sparse row matrix with comment ids and ngram
        features."""
        self.util_obj.out('constructing ngrams...')
        cv = self.count_vectorizer(s)
        str_list = cf[:]['text'].tolist()
        ngrams_m = cv.fit_transform(str_list)
        id_m = ss.lil_matrix((len(cf), 1))
        ngrams_csr = ss.hstack([id_m, ngrams_m]).tocsr()
        return ngrams_csr

    def split_mat(self, m, tr_df, te_df):
        """Splits the matrix into various datasets.
        m: matrix with all ngram features.
        tr_len: number of training examples.
        va_len: number of validation examples.
        te_len: number of testing examples.
        Returns three sparse matrices."""
        tr_len, te_len = len(tr_df), len(te_df)
        train_m = m[list(range(tr_len)), :]
        test_m = m[list(range(tr_len, tr_len + te_len)), :]
        return train_m, test_m

    def build_features(self, cf):
        """Selector to build features for the given domain.
        cf: comments dataframe.
        Returns dataframe containing content features."""
        if self.config_obj.domain == 'soundcloud':
            return self.soundcloud_features(cf)
        elif self.config_obj.domain == 'youtube':
            return self.youtube_features(cf)
        elif self.config_obj.domain == 'twitter':
            return self.twitter_features(cf)

    def soundcloud_features(self, cf):
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

    def youtube_features(self, cf):
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

    def twitter_features(self, cf):
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

    def build(self, tr_df, te_df):
        """Builds content features based on the text in the data.
        tr_df: training set dataframe.
        te_df: testing set dataframe.
        Returns ngram matrices for each dataset, content features dataframe,
                and a list of features created."""
        self.util_obj.start('building content features...')
        tr_m, te_m = None, None
        ngram_params = self.settings()
        coms_df = self.concat_coms(tr_df, te_df)
        c_df, feats_list = self.build_features(coms_df)
        if self.config_obj.ngrams:
            ngrams = self.ngrams(coms_df, ngram_params)
            tr_m, te_m = self.split_mat(ngrams, tr_df, te_df)
        self.util_obj.end()
        return tr_m, te_m, c_df, feats_list
