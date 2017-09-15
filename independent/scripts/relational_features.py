"""
This module creates relational features in sequential order of comments.
"""
import re
import numpy as np
import pandas as pd
from collections import defaultdict


class RelationalFeatures:
    """This class handles all operations to build relational features for
    each domain."""

    def __init__(self, config_obj, util_obj):
        """Initialize object dependencies."""

        self.config_obj = config_obj
        """User settings."""
        self.util_obj = util_obj
        """General utility methods."""

    # public
    def build(self, train_df, test_df):
        """Builds the relational features.
        train_df: training dataframe.
        test_df: testing dataframe.
        Returns relational features dataframe and list."""
        self.util_obj.start('building relational features...')
        bl, wl = self.settings()
        coms_df = self.concat_coms(train_df, test_df)
        features_df = self.build_features(coms_df, bl, wl)
        feature_list = features_df.columns.tolist()
        feature_list.remove('com_id')
        self.util_obj.end()
        return features_df, feature_list

    # private
    def settings(self):
        """Settings for relational features.
        Returns user blacklist limit, user whitelist lmiit."""
        blacklist, whitelist = 3, 10
        return blacklist, whitelist

    def concat_coms(self, train_df, test_df):
        """Removes labels from validation and test sets and concats datasets.
        train_df: training dataframe.
        test_df: testing dataframe.
        Returns concatenated datafrae."""
        test_df_copy = test_df.copy()
        test_df_copy['label'] = [np.nan for x in test_df_copy['label']]
        coms_df = pd.concat([train_df, test_df_copy])
        return coms_df

    def build_features(self, cf, bl, wl):
        """Selector to build features for the chosen domain.
        cf: comments dataframe.
        bl: blacklist threshold.
        wl: whitelist threshold.
        Returns dataframe of relational features."""
        if self.config_obj.domain == 'soundcloud':
            return self.soundcloud_features(cf)
        elif self.config_obj.domain == 'youtube':
            return self.youtube_features(cf, bl, wl)
        elif self.config_obj.domain == 'twitter':
            return self.twitter_features(cf)

    def soundcloud_features(self, coms_df):
        """Sequentially computes relational features in comments.
        coms_df: comments dataframe.
        Returns a dataframe of feature values for each comment."""

        # Data we want to keep track of and update for each comment.
        com_id_l = []
        user_c, user_l = defaultdict(int), []
        user_link_c, user_link_l = defaultdict(int), []
        user_spam_c, user_spam_l = defaultdict(int), []
        hub_c, hub_spam_c, hub_spam_l = defaultdict(int), defaultdict(int), []
        tr_c, tr_spam_c, track_spam_l = defaultdict(int), defaultdict(int), []
        util = self.util_obj

        # Generates relational features in sequential order.
        for r in coms_df.itertuples():
            com_id, u_id, tr_id = r[1], r[2], r[3]
            text, label = r[5], r[6],

            if self.config_obj.modified:
                text = r[7]

            # Add to lists.
            com_id_l.append(com_id)
            user_l.append(user_c[u_id])
            user_link_l.append(util.div0(user_link_c[u_id], user_c[u_id]))
            user_spam_l.append(util.div0(user_spam_c[u_id], user_c[u_id]))
            hub_spam_l.append(util.div0(hub_spam_c[text], hub_c[text]))
            track_spam_l.append(util.div0(tr_spam_c[tr_id], tr_c[tr_id]))

            # Update dictionaries.
            user_c[u_id] += 1
            hub_c[text] += 1
            tr_c[tr_id] += 1
            if 'http' in str(text):
                user_link_c[u_id] += 1
            if label == 1:
                user_spam_c[u_id] += 1
                hub_spam_c[text] += 1
                tr_spam_c[tr_id] += 1

        # Build features data frame.
        feats_dict = list(zip(com_id_l, user_l, user_link_l, user_spam_l,
                          hub_spam_l, track_spam_l))
        feats_df = pd.DataFrame(feats_dict)
        feats_df.columns = ['com_id', 'user_com_count', 'user_link_ratio',
                            'user_spam_ratio', 'text_spam_ratio',
                            'track_spam_ratio']
        return feats_df

    def youtube_features(self, coms_df, blacklist, whitelist):
        """Sequentially computes relational features in comments.
        coms_df: comments dataframe.
        blacklist: user spam post threshold.
        whitelist: user ham post threshold.
        Returns a dataframe of feature values for each comment."""

        # Data we want to keep track of and update for each comment.
        com_id_l = []
        user_c, user_l = defaultdict(int), []
        user_bl, user_wl = [], []
        user_len = defaultdict(lambda: [])
        user_max_l, user_min_l, user_mean_l = [], [], []
        user_spam_c, user_spam_l = defaultdict(int), []
        hub_c, hub_spam_c, hub_spam_l = defaultdict(int), defaultdict(int), []
        vid_c, vid_spam_c, vid_spam_l = defaultdict(int), defaultdict(int), []
        ho_c, ho_spam_c, ho_spam_l = defaultdict(int), defaultdict(int), []
        ment_c, ment_sp_c, ment_spam_l = defaultdict(int), defaultdict(int), []
        util = self.util_obj

        # Generates relational features in sequential order.
        ment_regex = re.compile(r"(@\w+)")
        for r in coms_df.itertuples():
            com_id, hour, vid_id = r[1], r[2][11:13], r[3]
            u_id, text, label = r[4], r[5], r[6]
            if self.config_obj.modified:
                text = r[7]
            mention = self.get_items(text, ment_regex)

            # Add to lists.
            com_id_l.append(com_id)
            user_l.append(user_c[u_id])
            user_spam_l.append(util.div0(user_spam_c[u_id], user_c[u_id]))
            hub_spam_l.append(util.div0(hub_spam_c[text], hub_c[text]))
            vid_spam_l.append(util.div0(vid_spam_c[vid_id], vid_c[vid_id]))
            ho_spam_l.append(util.div0(ho_spam_c[hour], ho_c[hour]))
            ment_spam_l.append(util.div0(ment_sp_c[mention], ment_c[mention]))

            user_ham_count = user_c[u_id] - user_spam_c[u_id]
            user_bl.append(1 if user_spam_c[u_id] > blacklist else 0)
            user_wl.append(1 if user_ham_count > whitelist else 0)

            user_lens = user_len[u_id]
            user_max, user_min, user_mean = 0, 0, 0
            if len(user_lens) > 0:
                user_max = max(user_lens)
                user_min = min(user_lens)
                user_mean = sum(user_lens) / float(len(user_lens))
            user_max_l.append(user_max)
            user_min_l.append(user_min)
            user_mean_l.append(user_mean)

            # Update dictionaries.
            ment_c[mention] += 1
            if label == 1:
                ment_sp_c[mention] += 1
            user_c[u_id] += 1
            hub_c[text] += 1
            vid_c[vid_id] += 1
            ho_c[hour] += 1
            user_len[u_id].append(len(text))
            if label == 1:
                user_spam_c[u_id] += 1
                hub_spam_c[text] += 1
                vid_spam_c[vid_id] += 1
                ho_spam_c[hour] += 1

        # Build features data frame.
        feats_dict = list(zip(com_id_l, user_l, user_bl, user_wl, user_max_l,
                          user_min_l, user_mean_l, user_spam_l, hub_spam_l,
                          vid_spam_l, ho_spam_l, ment_spam_l))
        feats_df = pd.DataFrame(feats_dict)
        feats_df.columns = ['com_id', 'user_com_count', 'user_blacklist',
                            'user_whitelist', 'user_max', 'user_min',
                            'user_mean', 'user_spam_ratio', 'text_spam_ratio',
                            'vid_spam_ratio', 'hour_spam_ratio',
                            'mention_spam_ratio']
        return feats_df

    def twitter_features(self, tweets_df):
        """Sequentially computes relational features in comments.
        coms_df: comments dataframe.
        Returns a dataframe of feature values for each comment."""

        # Data we want to keep track of and update for each tweet.
        tweet_c, tweet_l = defaultdict(int), []
        user_spam_c, user_spam_l = defaultdict(int), []
        link_c, link_l = defaultdict(int), []
        hash_c, hash_l = defaultdict(int), []
        ment_c, ment_l = defaultdict(int), []
        spam_c, hub_c, spam_l = defaultdict(int), defaultdict(int), []
        s_hash_c, h_hash_c, h_hash_l = defaultdict(int), defaultdict(int), []
        s_ment_c, h_ment_c, h_ment_l = defaultdict(int), defaultdict(int), []
        s_link_c, h_link_c, h_link_l = defaultdict(int), defaultdict(int), []
        tweet_id_l = []
        util = self.util_obj

        # Generates link_ratio, hashtag_ratio, mentions_ratio, spam_ratio.
        hash_regex = re.compile(r"(#\w+)")
        ment_regex = re.compile(r"(@\w+)")
        link_regex = re.compile(r"(http\w+)")
        for r in tweets_df.itertuples():
            tweet_id, u_id, text, label = r[1], r[2], r[3], r[4]
            if self.config_obj.modified:
                text = r[5]

            hashtag = self.get_items(text, hash_regex)
            mention = self.get_items(text, ment_regex)
            link = self.get_items(text, link_regex)

            # Add to lists.
            tweet_id_l.append(tweet_id)
            tweet_l.append(tweet_c[u_id])
            link_l.append(util.div0(link_c[u_id], tweet_c[u_id]))
            hash_l.append(util.div0(hash_c[u_id], tweet_c[u_id]))
            ment_l.append(util.div0(ment_c[u_id], tweet_c[u_id]))
            user_spam_l.append(util.div0(user_spam_c[u_id], tweet_c[u_id]))
            spam_l.append(util.div0(spam_c[text], hub_c[text]))
            h_hash_l.append(util.div0(s_hash_c[hashtag], h_hash_c[hashtag]))
            h_ment_l.append(util.div0(s_ment_c[mention], h_ment_c[mention]))
            h_link_l.append(util.div0(s_link_c[link], h_link_c[link]))

            # Update dictionaries.
            h_hash_c[hashtag] += 1
            if label == 1:
                s_hash_c[hashtag] += 1
            h_ment_c[mention] += 1
            if label == 1:
                s_ment_c[mention] += 1
            h_link_c[link] += 1
            if label == 1:
                s_link_c[link] += 1
            tweet_c[u_id] += 1
            hub_c[text] += 1
            if 'http' in text:
                link_c[u_id] += 1
            if '#' in text:
                hash_c[u_id] += 1
            if '@' in text:
                ment_c[u_id] += 1
            if label == 1:
                spam_c[text] += 1
                user_spam_c[u_id] += 1

        # Build features data frame.
        feats_dict = list(zip(tweet_id_l, tweet_l, link_l, hash_l,
                ment_l, user_spam_l, spam_l, h_hash_l, h_ment_l, h_link_l))
        feats_df = pd.DataFrame(feats_dict)
        feats_df.columns = ['com_id', 'user_com_count', 'user_link_ratio',
                'user_hashtag_ratio', 'user_mention_ratio', 'user_spam_ratio',
                'text_spam_ratio', 'hashtag_spam_ratio', 'mention_spam_ratio',
                'link_spam_ratio']
        feats_df = feats_df.drop([], axis=1)
        return feats_df

    def get_items(self, text, regex, str_form=True):
        """Method to extract hashtags from a string of text.
        text: text of the comment.
        regex: regex to extract items from the comment.
        str_form: concatenates list of items if True.
        Returns a string or list of item ids."""
        items = regex.findall(text)
        result = sorted([x.lower() for x in items])
        if str_form:
            result = ''.join(result)
        return result
