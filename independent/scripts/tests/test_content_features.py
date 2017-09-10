"""
Tests the content_features module.
"""
import unittest
import numpy as np
import pandas as pd
import scipy.sparse as ss
import mock
from sklearn.feature_extraction.text import CountVectorizer
from .context import content_features
from .context import config
from .context import test_utils as tu


class ContentFeaturesTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        self.test_obj = content_features.ContentFeatures(config_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        result = self.test_obj

        self.assertTrue(isinstance(result.config_obj, config.Config))

    def test_settings(self):
        setting_dict = {'stop_words': 'english', 'ngram_range': (3, 3),
                        'max_features': 10000, 'analyzer': 'char_wb',
                        'min_df': 6, 'max_df': 0.1, 'binary': True,
                        'vocabulary': None, 'dtype': np.int32}

        result = self.test_obj.settings()

        self.assertTrue(result == setting_dict)

    def test_concat_coms(self):
        train = ['a', 'b', None]
        val = ['a', 'b', None]
        test = ['a', 'b', None]
        train = pd.DataFrame({'text': train})
        val = pd.DataFrame({'text': val})
        test = pd.DataFrame({'text': test})

        result = self.test_obj.concat_coms(train, val, test)

        expected = pd.Series(['a', 'b', '', 'a', 'b', '', 'a', 'b', ''])
        self.assertTrue(len(result) == 9)
        self.assertTrue(result['text'].equals(expected))

    def test_count_vectorizer(self):
        setting_dict = {'stop_words': 'english', 'ngram_range': (3, 3),
                'max_features': 10000, 'analyzer': 'char_wb',
                'min_df': 6, 'max_df': 0.1, 'binary': True,
                'vocabulary': None, 'dtype': np.int32}

        result = self.test_obj.count_vectorizer(setting_dict)

        self.assertTrue(isinstance(result, CountVectorizer))

    def test_ngrams(self):
        setting_dict = {'stop_words': 'english', 'ngram_range': (3, 3),
                'max_features': 10000, 'analyzer': 'char_wb',
                'min_df': 6, 'max_df': 0.1, 'binary': True,
                'vocabulary': None, 'dtype': np.int32}
        matrix = mock.Mock(np.matrix)
        matrix.tocsr = mock.Mock(return_value='ngrams_csr')
        df = tu.sample_df(2)
        df['text'] = ['banana', 'orange']
        cv = mock.Mock(CountVectorizer)
        cv.fit_transform = mock.Mock(return_value='ngrams_m')
        self.test_obj.count_vectorizer = mock.Mock(return_value=cv)
        ss.lil_matrix = mock.Mock(return_value='id_m')
        ss.hstack = mock.Mock(return_value=matrix)

        result = self.test_obj.ngrams(df, setting_dict)

        self.test_obj.count_vectorizer.assert_called_with(setting_dict)
        cv.fit_transform.assert_called_with(['banana', 'orange'])
        ss.lil_matrix.assert_called_with((2, 1))
        ss.hstack.assert_called_with(['id_m', 'ngrams_m'])
        matrix.tocsr.assert_called()
        self.assertTrue(result == 'ngrams_csr')

    def test_split_mat(self):
        df1 = tu.sample_df(3)
        df2 = tu.sample_df(1)
        df3 = tu.sample_df(2)
        m = np.matrix([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])

        result = self.test_obj.split_mat(m, df1, df2, df3)

        self.assertTrue(result[0].shape == (3, 2))
        self.assertTrue(result[1].shape == (1, 2))
        self.assertTrue(result[2].shape == (2, 2))

    def test_build_features(self):
        self.test_obj.soundcloud_features = mock.Mock(return_value='feats')
        self.test_obj.youtube_features = mock.Mock()
        self.test_obj.twitter_features = mock.Mock()

        result = self.test_obj.build_features('df')

        self.assertTrue(result == 'feats')
        self.test_obj.soundcloud_features.assert_called_with('df')
        self.test_obj.youtube_features.assert_not_called()
        self.test_obj.twitter_features.assert_not_called()

    def test_soundcloud_features(self):
        df = tu.sample_df(2)
        df['text'] = ['banana', 'orange']

        result = self.test_obj.soundcloud_features(df)

        self.assertTrue(len(result[0] == 2))
        self.assertTrue(result[1] == ['com_num_chars', 'com_has_link'])

    def test_youtube_features(self):
        df = tu.sample_df(2)
        df['text'] = ['banana', 'orange']
        df['timestamp'] = ['2011-10-31 13:37:50', '2011-10-31 13:47:50']

        result = self.test_obj.youtube_features(df)

        self.assertTrue(len(result[0] == 2))
        self.assertTrue(result[1] == ['com_num_chars', 'com_weekday',
                'com_hour'])

    def test_twitter_features(self):
        df = tu.sample_df(2)
        df['text'] = ['bana@na', '#orange']
        df['timestamp'] = ['2011-10-31 13:37:50', '2011-10-31 13:47:50']

        result = self.test_obj.twitter_features(df)

        self.assertTrue(len(result[0] == 2))
        self.assertTrue(result[1] == ['com_num_chars', 'com_num_hashtags',
                'com_num_mentions', 'com_num_links', 'com_num_retweets'])

    def test_build(self):
        self.test_obj.config_obj.ngrams = True
        self.test_obj.settings = mock.Mock(return_value='ngram_params')
        self.test_obj.concat_coms = mock.Mock(return_value='coms_df')
        self.test_obj.build_features = mock.Mock(return_value=('df', 'feats'))
        self.test_obj.ngrams = mock.Mock(return_value='ngrams')
        self.test_obj.split_mat = mock.Mock(return_value=('trm', 'vam', 'tem'))

        result = self.test_obj.build('tr_df', 'va_df', 'te_df')

        self.test_obj.settings.assert_called()
        self.test_obj.concat_coms.assert_called_with('tr_df', 'va_df', 'te_df')
        self.test_obj.build_features.assert_called_with('coms_df')
        self.test_obj.ngrams.assert_called_with('coms_df', 'ngram_params')
        self.test_obj.split_mat.assert_called_with('ngrams', 'tr_df', 'va_df',
                'te_df')
        self.assertTrue(result == ('trm', 'vam', 'tem', 'df', 'feats'))


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(
            ContentFeaturesTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
