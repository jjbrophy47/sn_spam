"""
This module creates relational features in sequential order of messages.
"""
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
        df: messages dataframe.
        dset: dataset to test (e.g. 'val', 'test').
        fw: handle to write status updates to.
        Returns relational features dataframe and list."""
        self.util_obj.start('building relational features...', fw=fw)
        strip_df = self._strip_labels(df, dset=dset)
        feats_df, feats_list = self._build_features(strip_df)
        feats_list = [x for x in feats_list if x != 'com_id']
        self.util_obj.end(fw=fw)
        return feats_df, feats_list

    def _build_features(self, df):
        if self.config_obj.domain == 'soundcloud':
            features = ['com_id', 'user_msg_count', 'user_link_ratio',
                        'user_spam_ratio', 'text_spam_ratio',
                        'track_spam_ratio']
        elif self.config_obj.domain == 'youtube':
            features = ['com_id', 'user_msg_count', 'user_msg_len_max',
                        'user_msg_len_min', 'user_msg_len_mean']
        elif self.config_obj.domain == 'twitter':
            features = ['com_id', 'user_msg_count', 'user_link_ratio',
                        'user_hashtag_ratio', 'user_mention_ratio']
        elif self.config_obj.domain == 'toxic':
            features = []

        feats_df, feats_list = self._build_sequentially(df, features)
        return feats_df, feats_list

    def _build_features_dataframe(self, d):
        cols = []
        lists = []

        for k, v in d.items():
            cols.append(k)
            lists.append(v['list'])

        feats = list(zip(*lists))
        feats_df = pd.DataFrame(feats, columns=cols)
        feats_list = list(feats_df)

        return feats_df, feats_list

    def _build_sequentially(self, df, features):
        h, d = self._init_headers_and_super_dict(df, features)

        for r in df.itertuples():
            com_id, u_id, text, label = self._extract_column_values(r, h)
            self._update_relational(d, r, h, label)
            self._update_non_relational(d, features, com_id, u_id, text)

        feats_df, feats_list = self._build_features_dataframe(d)
        print(feats_df)
        return feats_df, feats_list

    def _extract_column_values(self, r, h):
        com_id, text = r[h['com_id']], r[h['text']]
        label = r[h['label']]
        if 'noisy_label' in h.keys():
            label = r[h['noisy_label']]
        u_id = r[h['user']]
        u_id = u_id[0] if type(u_id) == list else u_id
        return com_id, u_id, text, label

    def _init_headers_and_super_dict(self, df, features):
        headers = list(df)
        h = {h: i + 1 for i, h in enumerate(headers)}

        d = {}

        for feat in features:
            d[feat] = {'cnt': defaultdict(int), 'list': [],
                       'max': defaultdict(int), 'min': defaultdict(int),
                       'sum': defaultdict(int)}

        if self.config_obj.stacking > 0:
            for relation, group, group_id in self.config_obj.relations:
                d[group + '_spam_ratio'] = {'spam': defaultdict(float),
                                            'cnt': defaultdict(int),
                                            'list': []}
        return h, d

    def _strip_labels(self, df, dset='train'):
        df_copy = df.copy()
        if 'noisy_labels' in list(df_copy):
            df_copy['label'] = df_copy['noisy_labels']
        else:
            if dset != 'train':
                df_copy['label'] = [np.nan for x in df_copy['label']]
        return df_copy

    def _update_relational(self, d, row, headers, label):
        ut = self.util_obj

        if self.config_obj.stacking > 0:
            for relation, group, group_id in self.config_obj.relations:
                rd = d[group + '_spam_ratio']
                rel_ids = row[headers[group_id]]

                ratio = 0
                for rel_id in rel_ids:
                    ratio += ut.div0(rd['spam'][rel_id], rd['cnt'][rel_id])

                    rd['cnt'][rel_id] += 1
                    if label > 0:
                        rd['spam'][rel_id] += label

                rd['list'].append(ut.div0(ratio, len(rel_ids)))

    def _update_non_relational(self, d, keys, com_id, user_id, text):
        for key in keys:
            self._update_list(d, key, com_id, user_id)

        for key in keys:
            self._update_dict(d, key, user_id, text)

    def _update_list(self, d, k, com_id, uid):
        ut = self.util_obj
        umc = 'user_msg_count'

        if k == 'com_id':
            d[k]['list'].append(com_id)
        elif k == 'user_msg_count':
            d[k]['list'].append(d[k]['cnt'][uid])
        elif k == 'user_link_ratio':
            d[k]['list'].append(ut.div0(d[k]['cnt'][uid], d[umc]['cnt'][uid]))
        elif k == 'user_hashtag_ratio':
            d[k]['list'].append(ut.div0(d[k]['cnt'][uid], d[umc]['cnt'][uid]))
        elif k == 'user_mention_ratio':
            d[k]['list'].append(ut.div0(d[k]['cnt'][uid], d[umc]['cnt'][uid]))
        elif k == 'user_msg_len_max':
            d[k]['list'].append(d[k]['max'][uid])
        elif k == 'user_msg_len_min':
            d[k]['list'].append(d[k]['min'][uid])
        elif k == 'user_msg_len_mean':
            d[k]['list'].append(ut.div0(d[k]['sum'][uid], d[k]['cnt'][uid]))

    def _update_dict(self, d, k, uid, text):
        if k == 'user_msg_count':
            d[k]['cnt'][uid] += 1
        elif k == 'user_link_ratio':
            d[k]['cnt'][uid] += 1 if 'http' in text else 0
        elif k == 'user_hashtag_ratio':
            d[k]['cnt'][uid] += 1 if '#' in text else 0
        elif k == 'user_mention_ratio':
            d[k]['cnt'][uid] += 1 if '@' in text else 0
        elif k == 'user_msg_len_max':
            d[k]['max'][uid] = max(d[k]['max'][uid], len(text))
        elif k == 'user_msg_len_min':
            d[k]['min'][uid] = min(d[k]['min'][uid], len(text))
        elif k == 'user_msg_len_mean':
            d[k]['sum'][uid] += len(text)
            d[k]['cnt'][uid] += 1
