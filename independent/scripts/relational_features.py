"""
This module creates relational features in sequential order of messages.
"""
import re
import numpy as np
import pandas as pd
from collections import defaultdict


class RelationalFeatures:

    def __init__(self, config_obj, util_obj):
        self.config_obj = config_obj
        self.util_obj = util_obj

    # public
    def build(self, df, dset, fw=None):
        """Builds the relational features.
        train_df: training dataframe.
        test_df: testing dataframe.
        dset: dataset to test (e.g. 'val', 'test').
        Returns relational features dataframe and list."""
        self.util_obj.start('building relational features...', fw=fw)
        bl, wl = self.settings()
        strip_df = self.strip_labels(df, dset=dset)
        feats_df, feats_list = self.build_features(strip_df, bl, wl)
        feats_list = [x for x in feats_list if x != 'com_id']
        self.util_obj.end(fw=fw)
        return feats_df, feats_list

    # private
    def settings(self):
        blacklist, whitelist = 3, 10
        return blacklist, whitelist

    def strip_labels(self, df, dset='train'):
        df_copy = df.copy()
        if 'noisy_labels' in list(df_copy):
            df_copy['label'] = df_copy['noisy_labels']
        else:
            if dset != 'train':
                df_copy['label'] = [np.nan for x in df_copy['label']]
        return df_copy

    def build_features(self, cf, bl, wl, train_dicts=None):
        f_df, f_l = None, None

        if self.config_obj.domain == 'soundcloud':
            f_df, f_l = self.soundcloud(cf, train_dicts)
        elif self.config_obj.domain == 'youtube':
            f_df, f_l = self.youtube(cf, bl, wl, train_dicts)
        elif self.config_obj.domain == 'twitter':
            f_df, f_l = self.twitter(cf, train_dicts)
        elif self.config_obj.domain == 'toxic':
            f_df, f_l = self.toxic(cf, train_dicts)

        return f_df, f_l

    def soundcloud(self, coms_df):
        # Data we want to keep track of and update for each comment.
        com_id_l = []
        user_c, user_l = defaultdict(int), []
        user_link_c, user_link_l = defaultdict(int), []
        user_spam_c, user_spam_l = defaultdict(int), []
        hub_c, hub_spam_c, hub_spam_l = defaultdict(int), defaultdict(int), []
        tr_c, tr_spam_c, track_spam_l = defaultdict(int), defaultdict(int), []
        util = self.util_obj

        headers = list(coms_df)
        h = {h: i + 1 for i, h in enumerate(headers)}

        # Generates relational features in sequential order.
        for r in coms_df.itertuples():
            com_id = r[h['com_id']]
            text_id, label = r[h['text_id']], r[h['label']]

            for relation, group, group_id in self.config_obj.relations:
                rel_ids = r[h[group_id]]

                for rel_id in rel_ids:
                    pass

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

        if self.config_obj.stacking == 0:
            feats_df = feats_df.drop(['user_spam_ratio', 'text_spam_ratio',
                                      'track_spam_ratio'], axis=1)

        feats_l = list(feats_df)
        return feats_df, feats_l

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
        if self.config_obj.stacking == 0:
            feats_df = feats_df.drop(['user_spam_ratio', 'text_spam_ratio',
                                      'vid_spam_ratio', 'mention_spam_ratio'],
                                     axis=1)

        feats_l = list(feats_df)
        dicts = (user_c, user_len, user_spam_c, hub_c, hub_spam_c, vid_c,
                 vid_spam_c, ment_c, ment_sp_c)
        return feats_df, feats_l

    def twitter(self, df):
        ut = self.util_obj

        features = ['com_id', 'user_com_count', 'user_link_ratio',
                    'user_hashtag_ratio', 'user_mention_ratio',
                    'user_spam_ratio', 'text_spam_ratio',
                    'hashtag_spam_ratio', 'mention_spam_ratio',
                    'link_spam_ratio']

        h, d = self._init_headers_and_super_dict(df, features)

        for r in df.itertuples():
            com_id, text = r[h['com_id']], r[h['text']]
            label = r[h.get('noisy_label', None) if not None else h['label']]
            u_id = r[h['user_id']]
            u_id = u_id[0] if type(u_id) == list else u_id

            self._update_relational(d, r, h)
            self._update_non_relational(d, features, com_id, u_id, text)

        feats_df, feats_list = self._build_features_dataframe(d, features)
        return feats_df, feats_l

    def toxic(self, cf, train_dicts=None):
        feats_df = pd.DataFrame(cf['com_id'])
        feats_list = []
        return feats_df, feats_list


    # private
    def _build_features_dataframe(d):
        cols = []
        lists = []

        for k, v in d.items()
            cols.append(k)
            lists.append(v['list'])

        feats = [tuple(l[0] for l in lists)]
        feats_df = pd.DataFrame(feats, columns=cols)
        feats_list = list(feats_df)

        return feats_df, feats_list


    def _init_headers_and_super_dict(df, features):
        headers = list(df)
        h = {h: i + 1 for i, h in enumerate(headers)}

        d = {}

        for feat in features:
            d[feat] = {'cnt': defaultdict(int), 'list': defaultdict(list)}

        if self.config_obj.stacking > 0:
            for relation, group, group_id in self.config_obj.relations:
                d[group] = {'spam': defaultdict(float),
                            'cnt': defaultdict(int),
                            'list': defaultdict(list)}
        return h, d

    def _update_relational(d, row, headers):
        ut = self.util_obj

        if self.config_obj.stacking > 0:
            for relation, group, group_id in self.config_obj.relations:
                rd = d[group + '_spam_ratio']
                rel_ids = row[headers[group_id]]

                ratio = 0
                for rel_id in rel_ids:
                    ratio += ut.div0(rd['spam'][rel_id], rd['cnt'][rel_id])
                rd['list'].append(ut.div0(ratio, len(rel_ids)))

                if label > 0:
                    rd['spam'] += label
                rd['cnt'] += 1

    def _update_non_relational(d, keys, com_id, user_id, text):
        for key in keys:
            self._update_list(d, key, com_id, user_id)

        for key in keys:
            self._update_dict(d, key, user_id, text)


    def _update_list(d, keys, com_id, u_id):
        ut = self.util_obj
        umc = 'user_msg_count'

        if k == 'com_id':
            d[k]['list'].append(com_id)
        elif k == 'user_msg_count':
            d[k]['list'].append(d[k]['cnt'][u_id])
        elif k == 'user_link_ratio':
            d[k]['list'].append(ut.div0(d[k]['cnt'][u_id], d[umc]['cnt'][u_id])
        elif k == 'user_hashtag_ratio':
            d[k]['list'].append(ut.div0(d[k]['cnt'][u_id], d[umc]['cnt'][u_id])
        elif k == 'user_mention_ratio':
            d[k]['list'].append(ut.div0(d[k]['cnt'][u_id], d[umc]['cnt'][u_id])


    def _update_dict(d, k, u_id, text):
        if k == 'user_msg_count':
            d[umc]['cnt'][u_id] += 1
        elif k == 'user_link_ratio':
            d[k]['cnt'][u_id] += 1 if 'http' in text else 0
        elif k == 'user_hashtag_ratio':
            d[k]['cnt'][u_id] += 1 if '#' in text else 0
        elif k == 'user_mention_ratio':
            d[k]['cnt'][u_id] += 1 if '@' in text else 0
