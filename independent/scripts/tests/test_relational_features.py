"""
Tests the relational_features module.
"""
import re
import unittest
import numpy as np
import pandas as pd
import mock
from .context import relational_features
from .context import config
from .context import util
from .context import test_utils as tu


class RelationalFeaturesTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        util_obj = util.Util()
        self.test_obj = relational_features.RelationalFeatures(config_obj,
                util_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        result = self.test_obj

        self.assertTrue(isinstance(result.config_obj, config.Config))
        self.assertTrue(isinstance(result.util_obj, util.Util))

    def test_settings(self):
        result = self.test_obj.settings()

        self.assertTrue(result == (3, 10))

    def test_concat_coms(self):
        tr_df = tu.sample_df(10)
        te_df = tu.sample_df(5)
        tr_df.columns = ['com_id', 'label']
        te_df.columns = ['com_id', 'label']

        result = self.test_obj.concat_coms(tr_df, te_df)

        self.assertTrue(len(result) == 15)
        self.assertTrue(result['label'].sum() == 145)

    def test_build_features(self):
        self.test_obj.soundcloud_features = mock.Mock()
        self.test_obj.youtube_features = mock.Mock(return_value='feats')
        self.test_obj.twitter_features = mock.Mock()
        self.test_obj.config_obj.domain = 'youtube'

        result = self.test_obj.build_features('df', 7, 8)

        self.assertTrue(result == 'feats')
        self.test_obj.soundcloud_features.assert_not_called()
        self.test_obj.youtube_features.assert_called_with('df', 7, 8)
        self.test_obj.twitter_features.assert_not_called()

    def test_soundcloud_features(self):
        data = [[0, 1, 100, '', 'h', 0], [1, 2, 100, '', 't', 1],
                [2, 1, 100, '', 'h', 0], [3, 2, 102, '', 'b', 1]]
        df = pd.DataFrame(data, columns=['com_id', 'user_id', 'track_id',
                'timestamp', 'text', 'label'])

        result = self.test_obj.soundcloud_features(df)

        exp1 = pd.Series([0, 0, 1, 1])
        exp2 = pd.Series([0, 0, 0, 1.0])
        exp3 = pd.Series([0.0, 0.0, 0.0, 0.0])
        exp4 = pd.Series([0, 0, 0.5, 0])
        self.assertTrue(list(result) == ['com_id', 'user_com_count',
                'user_link_ratio', 'user_spam_ratio', 'text_spam_ratio',
                'track_spam_ratio'])
        self.assertTrue(len(result) == 4)
        self.assertTrue(result['user_com_count'].equals(exp1))
        self.assertTrue(result['user_spam_ratio'].equals(exp2))
        self.assertTrue(result['text_spam_ratio'].equals(exp3))
        self.assertTrue(result['track_spam_ratio'].equals(exp4))

    def test_youtube_features(self):
        data = [[0, '2011-10-31 13:37:50', 100, 1, 'h', 0],
                [1, '2011-10-31 13:37:50', 100, 2, 't', 1],
                [2, '2011-10-31 13:37:50', 100, 1, 'h', 0],
                [3, '2011-10-31 13:37:50', 102, 2, 'b', 1]]
        df = pd.DataFrame(data, columns=['com_id', 'user_id', 'track_id',
                'timestamp', 'text', 'label'])

        result = self.test_obj.youtube_features(df, 10, 10)

        exp1 = pd.Series([0, 0, 1, 1])
        exp2 = pd.Series([0.0, 0.0, 0.0, 1.0])
        exp3 = pd.Series([0.0, 0.0, 0.0, 0.0])
        exp4 = pd.Series([0.0, 0.0, 0.5, 0.0])
        exp5 = pd.Series([0.0, 0.0, 0.5, 0.333333])
        is_close5 = np.isclose(result['hour_spam_ratio'], exp5)
        exp5_bool = np.array([True, True, True, True])
        self.assertTrue(list(result) == ['com_id', 'user_com_count',
                'user_blacklist', 'user_whitelist', 'user_max', 'user_min',
                'user_mean', 'user_spam_ratio', 'text_spam_ratio',
                'vid_spam_ratio', 'hour_spam_ratio', 'mention_spam_ratio'])
        self.assertTrue(len(result) == 4)
        self.assertTrue(result['user_com_count'].equals(exp1))
        self.assertTrue(result['user_spam_ratio'].equals(exp2))
        self.assertTrue(result['text_spam_ratio'].equals(exp3))
        self.assertTrue(result['vid_spam_ratio'].equals(exp4))
        self.assertTrue(np.array_equal(is_close5, exp5_bool))

    def test_twitter_features(self):
        data = [[0, 1, '#h @f', 0], [1, 1, 't #h', 1],
                [2, 1, 't #h', 0], [3, 2, 'b #h', 1]]
        df = pd.DataFrame(data, columns=['com_id', 'user_id', 'text', 'label'])

        result = self.test_obj.twitter_features(df)

        exp1 = pd.Series([0, 1, 2, 0])
        exp2 = pd.Series([0.0, 0.0, 0.5, 0.0])
        exp3 = pd.Series([0.0, 0.0, 1.0, 0.0])
        exp4 = pd.Series([0.0, 0.0, 0.5, 0.3333333])
        is_close4 = np.isclose(result['hashtag_spam_ratio'], exp4)
        exp4_bool = np.array([True, True, True, True])
        self.assertTrue(list(result) == ['com_id', 'user_com_count',
                'user_link_ratio', 'user_hashtag_ratio', 'user_mention_ratio',
                'user_spam_ratio', 'text_spam_ratio', 'hashtag_spam_ratio',
                'mention_spam_ratio', 'link_spam_ratio'])
        self.assertTrue(len(result) == 4)
        self.assertTrue(result['user_com_count'].equals(exp1))
        self.assertTrue(result['user_spam_ratio'].equals(exp2))
        self.assertTrue(result['text_spam_ratio'].equals(exp3))
        self.assertTrue(np.array_equal(is_close4, exp4_bool))

    def test_get_items(self):
        hash_regex = re.compile(r"(#\w+)")

        result = self.test_obj.get_items('#ThIs is #sO cool', hash_regex)

        self.assertTrue(result == '#so#this')

    def test_build(self):
        df = tu.sample_df(10)
        self.test_obj.settings = mock.Mock(return_value=('bl', 'wl'))
        self.test_obj.concat_coms = mock.Mock(return_value='coms_df')
        self.test_obj.build_features = mock.Mock(return_value=df)

        result = self.test_obj.build('train_df', 'test_df')

        self.test_obj.settings.assert_called()
        self.test_obj.concat_coms.assert_called_with('train_df', 'test_df')
        self.test_obj.build_features.assert_called_with('coms_df', 'bl', 'wl')
        self.assertTrue(list(result[0]) == ['com_id', 'random'])
        self.assertTrue(len(result[0]) == 10)
        self.assertTrue(result[1] == ['random'])


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(
            RelationalFeaturesTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
