"""
Module that creates features based on the text of the comments.
"""
import os
import numpy as np
import pandas as pd
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

    # public
    def build(self, train_df, test_df, dset, fw=None):
        """Builds content features based on the text in the data.
        train_df: training set dataframe.
        test_df: testing set dataframe.
        dset: dataset (e.g. 'val', 'test').
        Returns ngram matrices for each dataset, content features dataframe,
                and a list of features created."""
        fold = self.config_obj.fold
        fn = 'train_' + dset + '_' + fold
        con_ext = '_content.pkl'
        ngram_ext = '_ngrams.npz'

        self.util_obj.start('building content features...', fw=fw)
        feats_f = self.define_file_folders()
        ngram_params = self.settings()
        tr_df, te_df, feats = self.basic(train_df, test_df, fn, con_ext,
                feats_f)
        coms_df = pd.concat([train_df, test_df])
        features_df = pd.concat([tr_df, te_df])
        tr_m, te_m = self.ngrams(coms_df, train_df, test_df, ngram_params,
                fn, ngram_ext, feats_f, fw=fw)
        self.util_obj.end(fw=fw)

        return tr_m, te_m, features_df, feats

    # private
    def define_file_folders(self):
        """Returns an absolute path to the features folder."""
        ind_dir = self.config_obj.ind_dir
        domain = self.config_obj.domain

        feats_f = ind_dir + 'output/' + domain + '/features/'
        if not os.path.exists(feats_f):
            os.makedirs(feats_f)
        return feats_f

    def settings(self):
        """Returns ngram settings. Strictly term frequency. Selects the
        top 10K features that appear most frequently. Features are then
        converted to binary for each document."""
        ngram_settings = {'stop_words': 'english', 'ngram_range': (3, 3),
                          'max_features': 10000, 'analyzer': 'char_wb',
                          'min_df': 1, 'max_df': 1.0, 'binary': True,
                          'vocabulary': None, 'dtype': np.int32}
        return ngram_settings

    def basic(self, train_df, test_df, fn, con_ext, feats_f):
        """Checks to see if there are already built features for this
                trainin set, and builds them if not.
        train_df: training set dataframe.
        test_df: test set dataframe.
        fn: filename of the saved content features.
        con_ext: extension of the filename.
        feats_f: features folder.
        Returns the training and test set feature dataframes with a list of
                feature names."""
        tr_df, _ = self.build_features(train_df)
        self.util_obj.save(tr_df, feats_f + fn + con_ext)
        te_df, feats_list = self.build_features(test_df)
        return tr_df, te_df, feats_list

    def ngrams(self, coms_df, train_df, test_df, ngram_params, fn, ngram_ext,
            feats_f, fw=None):
        """Loads ngram features for this training set if there are any,
                otherwise builds them.
        coms_df: combined train and test dataframes.
        train_df: training set dataframe.
        test_df: test set dataframe.
        ngram_params: parameters for selecting ngram features.
        fn: filename to save ngrams to.
        ngram_ext: filename extension.
        feats_f: features folder.
        Returns training and test set ngrams in matrix formats"""
        tr_m, te_m = None, None

        if self.config_obj.ngrams and self.config_obj.domain != 'ifwe':
            ngrams = self.build_ngrams(coms_df, ngram_params, fw=fw)
            self.util_obj.save_sparse(ngrams, feats_f + fn + ngram_ext)
            tr_m, te_m = self.split_mat(ngrams, train_df, test_df)
        return tr_m, te_m

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

    def build_ngrams(self, cf, s, fw=None):
        """Constructs ngrams based on the text in the comments.
        cf: comments dataframe.
        s: settings used for ngram construction.
        Returns a compressed sparse row matrix with comment ids and ngram
        features."""
        self.util_obj.write('constructing ngrams...', fw=fw)
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
        features_df, features_list = None, None

        if self.config_obj.domain != 'ifwe':
            cf['text'] = cf['text'].fillna('')

        if self.config_obj.domain == 'soundcloud':
            features_df, features_list = self.soundcloud(cf)
        elif self.config_obj.domain == 'youtube':
            features_df, features_list = self.youtube(cf)
        elif self.config_obj.domain == 'twitter':
            features_df, features_list = self.twitter(cf)
        elif self.config_obj.domain == 'toxic':
            features_df, features_list = self.toxic(cf)
        elif self.config_obj.domain == 'ifwe':
            features_df, features_list = self.ifwe(cf)
        elif self.config_obj.domain == 'yelp_hotel':
            features_df, features_list = self.yelp_hotel(cf)
        elif self.config_obj.domain == 'yelp_restaurant':
            features_df, features_list = self.yelp_restaurant(cf)

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
