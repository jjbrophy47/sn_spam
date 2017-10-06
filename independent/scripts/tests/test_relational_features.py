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

    @mock.patch('pandas.concat')
    def test_build_saved(self, mock_concat):
        df = tu.sample_df(10)
        self.test_obj.config_obj.saved = True
        self.test_obj.util_obj.start = mock.Mock()
        self.test_obj.define_file_folders = mock.Mock(return_value='f/')
        self.test_obj.settings = mock.Mock(return_value=('bl', 'wl'))
        self.test_obj.util_obj.load = mock.Mock()
        self.test_obj.util_obj.load.side_effect = ['tr_df', 'train_dicts']
        self.test_obj.strip_labels = mock.Mock(return_value='stripped')
        self.test_obj.build_features = mock.Mock(return_value=('te_df', 'l',
                ''))
        mock_concat.return_value = df
        self.test_obj.util_obj.end = mock.Mock()

        result = self.test_obj.build('train_df', 'test_df', 'test', fw='fw')

        exp_start = 'building relational features...'
        exp_load = [mock.call('f/save_train_test_1_rfeats.pkl'),
                mock.call('f/save_train_test_1_rdicts.pkl')]
        self.assertTrue(list(result[0]) == ['com_id', 'random'])
        self.assertTrue(len(result[0]) == 10)
        self.assertTrue(result[1] == ['l'])
        self.test_obj.util_obj.start.assert_called_with(exp_start, fw='fw')
        self.test_obj.define_file_folders.assert_called()
        self.test_obj.settings.assert_called()
        self.assertTrue(self.test_obj.util_obj.load.call_args_list == exp_load)
        self.test_obj.strip_labels.assert_called_with('test_df')
        self.test_obj.build_features.assert_called_with('stripped', 'bl', 'wl',
                'train_dicts')
        mock_concat.assert_called_with(['tr_df', 'te_df'])
        self.test_obj.util_obj.end.assert_called_with(fw='fw')

    @mock.patch('pandas.concat')
    def test_build_not_saved(self, mock_concat):
        df = tu.sample_df(10)
        self.test_obj.config_obj.saved = False
        self.test_obj.util_obj.start = mock.Mock()
        self.test_obj.define_file_folders = mock.Mock(return_value='f/')
        self.test_obj.settings = mock.Mock(return_value=('bl', 'wl'))
        self.test_obj.util_obj.load = mock.Mock()
        self.test_obj.util_obj.save = mock.Mock()
        self.test_obj.build_features = mock.Mock()
        self.test_obj.build_features.side_effect = [('tr', '', 'td'),
                ('te', 'l', '')]
        self.test_obj.strip_labels = mock.Mock(return_value='stripped')
        mock_concat.return_value = df
        self.test_obj.util_obj.end = mock.Mock()

        result = self.test_obj.build('train_df', 'test_df', 'test', fw='fw')

        exp_start = 'building relational features...'
        exp_save = [mock.call('tr', 'f/train_test_1_rfeats.pkl'),
                mock.call('td', 'f/train_test_1_rdicts.pkl')]
        exp_build = [mock.call('train_df', 'bl', 'wl'),
                mock.call('stripped', 'bl', 'wl', 'td')]
        self.assertTrue(list(result[0]) == ['com_id', 'random'])
        self.assertTrue(len(result[0]) == 10)
        self.assertTrue(result[1] == ['l'])
        self.test_obj.util_obj.start.assert_called_with(exp_start, fw='fw')
        self.test_obj.define_file_folders.assert_called()
        self.test_obj.settings.assert_called()
        self.test_obj.util_obj.load.assert_not_called()
        self.assertTrue(self.test_obj.util_obj.save.call_args_list == exp_save)
        self.test_obj.strip_labels.assert_called_with('test_df')
        self.assertTrue(self.test_obj.build_features.call_args_list ==
                exp_build)
        mock_concat.assert_called_with(['tr', 'te'])
        self.test_obj.util_obj.end.assert_called_with(fw='fw')

    def test_define_file_folders(self):
        result = self.test_obj.define_file_folders()

        self.assertTrue(result == 'ind/output/soundcloud/features/')

    def test_settings(self):
        result = self.test_obj.settings()

        self.assertTrue(result == (3, 10))

    def test_strip_labels(self):
        df = tu.sample_df(2)
        df_copy = tu.sample_df(2)
        df_copy.columns = ['com_id', 'label']
        df.copy = mock.Mock(return_value=df_copy)

        result = self.test_obj.strip_labels(df)

        self.assertTrue(list(result['com_id']) == [0, 1])
        self.assertTrue(np.isnan(result['label'].sum()))

    def test_build_features(self):
        feats = ('f', 'd', 'l')
        self.test_obj.soundcloud = mock.Mock()
        self.test_obj.youtube = mock.Mock(return_value=feats)
        self.test_obj.twitter = mock.Mock()
        self.test_obj.ifwe = mock.Mock()
        self.test_obj.yelp_hotel = mock.Mock()
        self.test_obj.yelp_restaurant = mock.Mock()
        self.test_obj.config_obj.domain = 'youtube'

        result = self.test_obj.build_features('df', 7, 8, train_dicts='td')

        self.assertTrue(result == ('f', 'd', 'l'))
        self.test_obj.soundcloud.assert_not_called()
        self.test_obj.youtube.assert_called_with('df', 7, 8, 'td')
        self.test_obj.twitter.assert_not_called()
        self.test_obj.ifwe.assert_not_called()
        self.test_obj.yelp_hotel.assert_not_called()
        self.test_obj.yelp_restaurant.assert_not_called()

    def test_soundcloud(self):
        data = [[0, 1, 100, '', 'h', 0], [1, 2, 100, '', 't', 1],
                [2, 1, 100, '', 'h', 0], [3, 2, 102, '', 'b', 1]]
        df = pd.DataFrame(data, columns=['com_id', 'user_id', 'track_id',
                'timestamp', 'text', 'label'])
        self.test_obj.config_obj.pseudo = True

        result = self.test_obj.soundcloud(df)

        exp1 = pd.Series([0, 0, 1, 1])
        exp2 = pd.Series([0, 0, 0, 1.0])
        exp3 = pd.Series([0.0, 0.0, 0.0, 0.0])
        exp4 = pd.Series([0, 0, 0.5, 0])
        self.assertTrue(list(result[0]) == ['com_id', 'user_com_count',
                'user_link_ratio', 'user_spam_ratio', 'text_spam_ratio',
                'track_spam_ratio'])
        self.assertTrue(len(result[0]) == 4)
        self.assertTrue(result[0]['user_com_count'].equals(exp1))
        self.assertTrue(result[0]['user_spam_ratio'].equals(exp2))
        self.assertTrue(result[0]['text_spam_ratio'].equals(exp3))
        self.assertTrue(result[0]['track_spam_ratio'].equals(exp4))
        self.assertTrue(len(result[1]) == 6)
        self.assertTrue(len(result[2]) == 7)

    def test_youtube(self):
        data = [[0, '2011-10-31 13:37:50', 100, 1, 'h', 0],
                [1, '2011-10-31 13:37:50', 100, 2, 't', 1],
                [2, '2011-10-31 13:37:50', 100, 1, 'h', 0],
                [3, '2011-10-31 13:37:50', 102, 2, 'b', 1]]
        df = pd.DataFrame(data, columns=['com_id', 'user_id', 'track_id',
                'timestamp', 'text', 'label'])
        self.test_obj.config_obj.pseudo = True

        result = self.test_obj.youtube(df, 10, 10)

        exp1 = pd.Series([0, 0, 1, 1])
        exp2 = pd.Series([0.0, 0.0, 0.0, 1.0])
        exp3 = pd.Series([0.0, 0.0, 0.0, 0.0])
        exp4 = pd.Series([0.0, 0.0, 0.5, 0.0])
        self.assertTrue(list(result[0]) == ['com_id', 'user_com_count',
                'user_blacklist', 'user_whitelist', 'user_max', 'user_min',
                'user_mean', 'user_spam_ratio', 'text_spam_ratio',
                'vid_spam_ratio', 'mention_spam_ratio'])
        self.assertTrue(len(result[0]) == 4)
        self.assertTrue(result[0]['user_com_count'].equals(exp1))
        self.assertTrue(result[0]['user_spam_ratio'].equals(exp2))
        self.assertTrue(result[0]['text_spam_ratio'].equals(exp3))
        self.assertTrue(result[0]['vid_spam_ratio'].equals(exp4))
        self.assertTrue(len(result[1]) == 11)
        self.assertTrue(len(result[2]) == 9)

    def test_twitter(self):
        data = [[0, 1, '#h @f', 0], [1, 1, 't #h', 1],
                [2, 1, 't #h', 0], [3, 2, 'b #h', 1]]
        df = pd.DataFrame(data, columns=['com_id', 'user_id', 'text', 'label'])
        self.test_obj.config_obj.pseudo = True

        result = self.test_obj.twitter(df)

        exp1 = pd.Series([0, 1, 2, 0])
        exp2 = pd.Series([0.0, 0.0, 0.5, 0.0])
        exp3 = pd.Series([0.0, 0.0, 1.0, 0.0])
        exp4 = pd.Series([0.0, 0.0, 0.5, 0.3333333])
        is_close4 = np.isclose(result[0]['hashtag_spam_ratio'], exp4)
        exp4_bool = np.array([True, True, True, True])
        self.assertTrue(list(result[0]) == ['com_id', 'user_com_count',
                'user_link_ratio', 'user_hashtag_ratio', 'user_mention_ratio',
                'user_spam_ratio', 'text_spam_ratio', 'hashtag_spam_ratio',
                'mention_spam_ratio', 'link_spam_ratio'])
        self.assertTrue(len(result[0]) == 4)
        self.assertTrue(result[0]['user_com_count'].equals(exp1))
        self.assertTrue(result[0]['user_spam_ratio'].equals(exp2))
        self.assertTrue(result[0]['text_spam_ratio'].equals(exp3))
        self.assertTrue(np.array_equal(is_close4, exp4_bool))
        self.assertTrue(len(result[1]) == 10)
        self.assertTrue(len(result[2]) == 9)

    @mock.patch('pandas.DataFrame')
    def test_ifwe(self, mock_df):
        df = tu.sample_df(10)
        mock_df.return_value = 'feats_df'

        result = self.test_obj.ifwe(df)

        exp_list = ['0_0', '0_1', '0_2', '0_3', '0_4', '0_5', '0_6', '0_7',
                '1_0', '1_1', '1_2', '1_3', '1_4', '1_5', '1_6', '1_7', '2_0',
                '2_1', '2_2', '2_3', '2_4', '2_5', '2_6', '2_7', '3_0', '3_1',
                '3_2', '3_3', '3_4', '3_5', '3_6', '3_7', '4_0', '4_1', '4_2',
                '4_3', '4_4', '4_5', '4_6', '4_7', '5_0', '5_1', '5_2', '5_3',
                '5_4', '5_5', '5_6', '5_7', '6_0', '6_1', '6_2', '6_3', '6_4',
                '6_5', '6_6', '6_7', '7_0', '7_1', '7_2', '7_3', '7_4', '7_5',
                '7_6', '7_7']
        self.assertTrue(result == ('feats_df', exp_list, ()))
        mock_df.assert_called_with(df['com_id'])

    def test_yelp_hotel(self):
        data = [[0, 0, 1, 1, 'text', 0], [1, 0, 1, 2, 'texting', 1],
                [2, 0, 1, 1, 'text2', 0], [3, 0, 1, 2, 'text2', 1]]
        df = pd.DataFrame(data, columns=['com_id', 'timestamp', 'user_id',
                'hotel_id', 'text', 'label'])
        self.test_obj.config_obj.pseudo = True

        result = self.test_obj.yelp_hotel(df)

        exp1 = pd.Series([0.0, 0.0, 0.5, 0.3333333])
        is_close1 = np.isclose(result[0]['user_spam_ratio'], exp1)
        exp1_bool = np.array([True, True, True, True])
        exp2 = pd.Series([0.0, 0.0, 0.0, 1.0])
        exp3 = pd.Series([0.0, 0.0, 0.0, 0.0])
        self.assertTrue(list(result[0]) == ['com_id', 'user_spam_ratio',
                'hotel_spam_ratio', 'text_spam_ratio'])
        self.assertTrue(result[0]['hotel_spam_ratio'].equals(exp2))
        self.assertTrue(result[0]['text_spam_ratio'].equals(exp3))
        self.assertTrue(np.array_equal(is_close1, exp1_bool))
        self.assertTrue(len(result[1]) == 23)
        self.assertTrue(len(result[2]) == 6)

    def test_yelp_restaurant(self):
        data = [[0, 0, 1, 1, 'text', 0], [1, 0, 1, 2, 'text', 1],
                [2, 0, 1, 1, 'text', 0], [3, 0, 1, 2, 'text2', 1]]
        df = pd.DataFrame(data, columns=['com_id', 'timestamp', 'user_id',
                'rest_id', 'text', 'label'])
        self.test_obj.config_obj.pseudo = True

        result = self.test_obj.yelp_restaurant(df)

        exp1 = pd.Series([0.0, 0.0, 0.5, 0.3333333])
        is_close1 = np.isclose(result[0]['user_spam_ratio'], exp1)
        exp1_bool = np.array([True, True, True, True])
        exp2 = pd.Series([0.0, 0.0, 0.0, 1.0])
        exp3 = pd.Series([0.0, 0.0, 0.5, 0.0])
        self.assertTrue(list(result[0]) == ['com_id', 'user_spam_ratio',
                'rest_spam_ratio', 'text_spam_ratio'])
        self.assertTrue(result[0]['rest_spam_ratio'].equals(exp2))
        self.assertTrue(result[0]['text_spam_ratio'].equals(exp3))
        self.assertTrue(np.array_equal(is_close1, exp1_bool))
        self.assertTrue(len(result[1]) == 39)
        self.assertTrue(len(result[2]) == 6)

    def test_get_items(self):
        hash_regex = re.compile(r"(#\w+)")

        result = self.test_obj.get_items('#ThIs is #sO cool', hash_regex)

        self.assertTrue(result == '#so#this')


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(
            RelationalFeaturesTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
