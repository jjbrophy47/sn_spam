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
    def build(self, train_df, test_df, dset, fw=None):
        """Builds the relational features.
        train_df: training dataframe.
        test_df: testing dataframe.
        dset: dataset to test (e.g. 'val', 'test').
        Returns relational features dataframe and list."""
        self.util_obj.start('building relational features...', fw=fw)
        bl, wl = self.settings()
        tr_df, _, train_dicts = self.build_features(train_df, bl, wl)
        test_strip_df = self.strip_labels(test_df)
        te_df, l, _ = self.build_features(test_strip_df, bl, wl, train_dicts)
        features_df = pd.concat([tr_df, te_df])
        l = [x for x in l if x != 'com_id']
        self.util_obj.end(fw=fw)
        return features_df, l

    # private
    def settings(self):
        """Settings for relational features.
        Returns user blacklist limit, user whitelist lmiit."""
        blacklist, whitelist = 3, 10
        return blacklist, whitelist

    def strip_labels(self, df):
        """Replaces the labels with NaNs, unless there are predicted labes
        df: dataframe with labels.
        Returns dataframe with replaced labels."""
        df_copy = df.copy()
        if 'noisy_labels' in list(df_copy):
            df_copy['label'] = df_copy['noisy_labels']
        else:
            df_copy['label'] = [np.nan for x in df_copy['label']]
        return df_copy

    def build_features(self, cf, bl, wl, train_dicts=None):
        """Selector to build features for the chosen domain.
        cf: comments dataframe.
        bl: blacklist threshold.
        wl: whitelist threshold.
        Returns dataframe of relational features, list of feature names,
                and a dictionaries used to build the features."""
        f_df, f_l, f_d = None, None, None

        if self.config_obj.domain == 'soundcloud':
            f_df, f_l, f_d = self.soundcloud(cf, train_dicts)
        elif self.config_obj.domain == 'youtube':
            f_df, f_l, f_d = self.youtube(cf, bl, wl, train_dicts)
        elif self.config_obj.domain == 'twitter':
            f_df, f_l, f_d = self.twitter(cf, train_dicts)

        return f_df, f_l, f_d

    def soundcloud(self, coms_df, train_dicts=None):
        """Sequentially computes relational features in comments.
        coms_df: comments dataframe.
        train_dicts: filled in dicts from the training data.
        Returns a dataframe of feature values for each comment."""

        # Data we want to keep track of and update for each comment.
        com_id_l = []
        user_c, user_l = defaultdict(int), []
        user_link_c, user_link_l = defaultdict(int), []
        user_spam_c, user_spam_l = defaultdict(int), []
        hub_c, hub_spam_c, hub_spam_l = defaultdict(int), defaultdict(int), []
        tr_c, tr_spam_c, track_spam_l = defaultdict(int), defaultdict(int), []
        util = self.util_obj

        if train_dicts is not None:
            user_c, user_link_c, user_spam_c, hub_c, hub_spam_c, tr_c,\
                tr_spam_c = train_dicts

        # Generates relational features in sequential order.
        for r in coms_df.itertuples():
            com_id, u_id, tr_id = r[1], r[2], r[3]
            text, label = r[5], r[6],
            text_id = text
            if self.config_obj.modified:
                text_id = r[7]

            # Add to lists.
            com_id_l.append(com_id)
            user_l.append(user_c[u_id])
            user_link_l.append(util.div0(user_link_c[u_id], user_c[u_id]))
            user_spam_l.append(util.div0(user_spam_c[u_id], user_c[u_id]))
            hub_spam_l.append(util.div0(hub_spam_c[text_id], hub_c[text_id]))
            track_spam_l.append(util.div0(tr_spam_c[tr_id], tr_c[tr_id]))

            # Update dictionaries.
            user_c[u_id] += 1
            hub_c[text_id] += 1
            tr_c[tr_id] += 1
            if 'http' in str(text):
                user_link_c[u_id] += 1
            if label > 0:
                user_spam_c[u_id] += label
                hub_spam_c[text_id] += label
                tr_spam_c[tr_id] += label

        # Build features data frame.
        feats_dict = list(zip(com_id_l, user_l, user_link_l, user_spam_l,
                          hub_spam_l, track_spam_l))
        feats_df = pd.DataFrame(feats_dict)
        feats_df.columns = ['com_id', 'user_com_count', 'user_link_ratio',
                            'user_spam_ratio', 'text_spam_ratio',
                            'track_spam_ratio']

        if not self.config_obj.pseudo:
            feats_df = feats_df.drop(['user_spam_ratio', 'text_spam_ratio',
                    'track_spam_ratio'], axis=1)

        feats_l = list(feats_df)
        dicts = (user_c, user_link_c, user_spam_c, hub_c, hub_spam_c, tr_c,
                tr_spam_c)
        return feats_df, feats_l, dicts

    def youtube(self, coms_df, blacklist, whitelist,
            train_dicts=None):
        """Sequentially computes relational features in comments.
        coms_df: comments dataframe.
        blacklist: user spam post threshold.
        whitelist: user ham post threshold.
        train_dicts: filled in dicts from the training data.
        Returns a dataframe of feature values for each comment."""

        # Data we want to keep track of and update for each comment.
        com_id_l = []
        user_c, user_l = defaultdict(int), []
        user_bl, user_wl = [], []
        user_len = defaultdict(list)
        user_max_l, user_min_l, user_mean_l = [], [], []
        user_spam_c, user_spam_l = defaultdict(int), []
        hub_c, hub_spam_c, hub_spam_l = defaultdict(int), defaultdict(int), []
        vid_c, vid_spam_c, vid_spam_l = defaultdict(int), defaultdict(int), []
        ment_c, ment_sp_c, ment_spam_l = defaultdict(int), defaultdict(int), []
        util = self.util_obj

        if train_dicts is not None:
            user_c, user_len, user_spam_c, hub_c, hub_spam_c, vid_c,\
                vid_spam_c, ment_c, ment_sp_c = train_dicts

        # Generates relational features in sequential order.
        ment_regex = re.compile(r"(@\w+)")
        for r in coms_df.itertuples():
            com_id, vid_id = r[1], r[3]
            u_id, text, label = r[4], r[5], r[6]
            text_id = text
            if self.config_obj.modified:
                text_id = r[7]
            mention = self.get_items(text, ment_regex)

            # Add to lists.
            com_id_l.append(com_id)
            user_l.append(user_c[u_id])
            user_spam_l.append(util.div0(user_spam_c[u_id], user_c[u_id]))
            hub_spam_l.append(util.div0(hub_spam_c[text_id], hub_c[text_id]))
            vid_spam_l.append(util.div0(vid_spam_c[vid_id], vid_c[vid_id]))
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
            if label > 0:
                ment_sp_c[mention] += label
            user_c[u_id] += 1
            hub_c[text_id] += 1
            vid_c[vid_id] += 1
            user_len[u_id].append(len(text))
            if label > 0:
                user_spam_c[u_id] += label
                hub_spam_c[text_id] += label
                vid_spam_c[vid_id] += label

        # Build features data frame.
        feats_dict = list(zip(com_id_l, user_l, user_bl, user_wl, user_max_l,
                          user_min_l, user_mean_l, user_spam_l, hub_spam_l,
                          vid_spam_l, ment_spam_l))
        feats_df = pd.DataFrame(feats_dict)
        feats_df.columns = ['com_id', 'user_com_count', 'user_blacklist',
                            'user_whitelist', 'user_max', 'user_min',
                            'user_mean', 'user_spam_ratio', 'text_spam_ratio',
                            'vid_spam_ratio', 'mention_spam_ratio']
        if not self.config_obj.pseudo:
            feats_df = feats_df.drop(['user_spam_ratio', 'text_spam_ratio',
                    'vid_spam_ratio', 'mention_spam_ratio'], axis=1)

        feats_l = list(feats_df)
        dicts = (user_c, user_len, user_spam_c, hub_c, hub_spam_c, vid_c,
                vid_spam_c, ment_c, ment_sp_c)
        return feats_df, feats_l, dicts

    def twitter(self, tweets_df, train_dicts=None):
        """Sequentially computes relational features in comments.
        coms_df: comments dataframe.
        train_dicts: filled in dicts from the training data.
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

        if train_dicts is not None:
            tweet_c, user_spam_c, link_c, hash_c, ment_c, spam_c, s_hash_c,\
                s_ment_c, s_link_c = train_dicts

        # Generates link_ratio, hashtag_ratio, mentions_ratio, spam_ratio.
        hash_regex = re.compile(r"(#\w+)")
        ment_regex = re.compile(r"(@\w+)")
        link_regex = re.compile(r"(http\w+)")
        for r in tweets_df.itertuples():
            tweet_id, u_id, text, label = r[1], r[2], r[3], r[4]
            text_id = text
            if self.config_obj.modified:
                text_id = r[5]

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
            spam_l.append(util.div0(spam_c[text_id], hub_c[text_id]))
            h_hash_l.append(util.div0(s_hash_c[hashtag], h_hash_c[hashtag]))
            h_ment_l.append(util.div0(s_ment_c[mention], h_ment_c[mention]))
            h_link_l.append(util.div0(s_link_c[link], h_link_c[link]))

            # Update dictionaries.
            h_hash_c[hashtag] += 1
            if label > 0:
                s_hash_c[hashtag] += label
            h_ment_c[mention] += 1
            if label > 0:
                s_ment_c[mention] += label
            h_link_c[link] += 1
            if label > 0:
                s_link_c[link] += label
            tweet_c[u_id] += 1
            hub_c[text] += 1
            if 'http' in text:
                link_c[u_id] += 1
            if '#' in text:
                hash_c[u_id] += 1
            if '@' in text:
                ment_c[u_id] += 1
            if label > 0:
                spam_c[text_id] += label
                user_spam_c[u_id] += label

        # Build features data frame.
        feats_dict = list(zip(tweet_id_l, tweet_l, link_l, hash_l,
                ment_l, user_spam_l, spam_l, h_hash_l, h_ment_l, h_link_l))
        feats_df = pd.DataFrame(feats_dict)
        feats_df.columns = ['com_id', 'user_com_count', 'user_link_ratio',
                'user_hashtag_ratio', 'user_mention_ratio', 'user_spam_ratio',
                'text_spam_ratio', 'hashtag_spam_ratio', 'mention_spam_ratio',
                'link_spam_ratio']
        if not self.config_obj.pseudo:
            feats_df = feats_df.drop(['user_spam_ratio', 'text_spam_ratio',
                    'hashtag_spam_ratio', 'mention_spam_ratio',
                    'link_spam_ratio'], axis=1)

        feats_l = list(feats_df)
        dicts = (tweet_c, user_spam_c, link_c, hash_c, ment_c, spam_c,
                s_hash_c, s_ment_c, s_link_c)
        return feats_df, feats_l, dicts

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
