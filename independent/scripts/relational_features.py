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
    def build(self, df, dset):
        """Builds the relational features.
        df: messages dataframe.
        dset: dataset to test (e.g. 'val', 'test').
        fw: handle to write status updates to.
        Returns relational features dataframe and list."""
        self.util_obj.out('building relational features...')
        strip_df = self._strip_labels(df, dset=dset)
        feats_df, feats_list = self._build_features(strip_df)
        feats_list = [x for x in feats_list if x != 'com_id']
        return feats_df, feats_list

    def _build_features(self, df):
        featureset = self.config_obj.featureset

        if self.config_obj.domain == 'adclicks':
            features = ['com_id', 'ip_click_count', 'app_click_count',
                        'channel_click_count', 'channel_ip_click_ratio',
                        'app_ip_click_ratio', 'channel_ip_hour_ratio',
                        'app_ip_hour_ratio']
            # features = ['com_id']
        elif self.config_obj.domain == 'soundcloud':
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
            features = ['com_id']

        if not any(x in featureset for x in ['sequential', 'all']):
            features = ['com_id']

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
            com_id, v = self._extract_column_values(r, h)
            self._update_relational(d, r, h, v)
            self._update_non_relational(d, features, com_id, v)

        feats_df, feats_list = self._build_features_dataframe(d)
        return feats_df, feats_list

    def _extract_column_values(self, r, h):
        v = {}
        com_id = r[h['com_id']]

        if 'label' in h:
            v['label'] = r[h['label']]
            if 'noisy_label' in h.keys():
                v['label'] = r[h['noisy_label']]

        if 'text' in h:
            v['text'] = r[h['text']]
        if 'user' in h:
            u_id = r[h['user']]
            u_id = u_id[0] if type(u_id) == list else u_id
            v['user'] = u_id
        if 'ip' in h:
            v['ip'] = r[h['ip']]
        if 'app' in h:
            v['app'] = r[h['app']]
        if 'channel' in h:
            v['channel'] = r[h['channel']]
        if 'click_time' in h:
            time = pd.to_datetime(r[h['click_time']])
            v['day'] = time.dayofweek
            v['hour'] = time.hour
            v['minute'] = time.minute

        return com_id, v

    def _init_headers_and_super_dict(self, df, features):
        domain = self.config_obj.domain
        label_name = 'spam' if domain != 'adclicks' else 'attribution'

        headers = list(df)
        h = {h: i + 1 for i, h in enumerate(headers)}

        d = {}

        for feat in features:
            d[feat] = {'cnt': defaultdict(int), 'list': [],
                       'max': defaultdict(int), 'min': defaultdict(int),
                       'sum': defaultdict(int)}

        if self.config_obj.stacking > 0:
            for relation, group, group_id in self.config_obj.relations:
                key = group + '_' + label_name + '_ratio'
                d[key] = {'label': defaultdict(float), 'cnt': defaultdict(int),
                          'list': []}
        return h, d

    def _strip_labels(self, df, dset='train'):
        df_copy = df.copy()
        if 'noisy_labels' in list(df_copy):
            df_copy['label'] = df_copy['noisy_labels']
        elif 'label' in list(df_copy):
            if dset != 'train':
                df_copy['label'] = [np.nan for x in df_copy['label']]
        return df_copy

    def _update_relational(self, d, row, headers, v):
        ut = self.util_obj
        domain = self.config_obj.domain
        label_name = 'spam' if domain != 'adclicks' else 'attribution'
        label = v.get('label', None)

        if self.config_obj.stacking > 0:
            for relation, group, group_id in self.config_obj.relations:
                rd = d[group + '_' + label_name + '_ratio']
                rel_ids = row[headers[group_id]]

                ratio = 0
                for rel_id in rel_ids:
                    ratio += ut.div0(rd['label'][rel_id], rd['cnt'][rel_id])

                    rd['cnt'][rel_id] += 1
                    if label > 0:
                        rd['label'][rel_id] += label

                rd['list'].append(ut.div0(ratio, len(rel_ids)))

    def _update_non_relational(self, d, keys, com_id, v):
        for key in keys:
            self._update_list(d, key, com_id, v)

        for key in keys:
            self._update_dict(d, key, v)

    def _update_list(self, d, k, com_id, v):
        ut = self.util_obj
        umc = 'user_msg_count'
        ccc = 'channel_click_count'
        acc = 'app_click_count'
        uid = v.get('user', None)
        ipid = v.get('ip', None)
        appid = v.get('app', None)
        channelid = v.get('channel', None)
        dayid = v.get('day', None)
        hourid = v.get('hour', None)
        minuteid = v.get('minute', None)

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
        elif k == 'ip_click_count':
            d[k]['list'].append(d[k]['cnt'][ipid])
        elif k == 'app_click_count':
            d[k]['list'].append(d[k]['cnt'][appid])
        elif k == 'channel_click_count':
            d[k]['list'].append(d[k]['cnt'][channelid])
        elif k == 'channel_ip_click_ratio':
            cipid = str(channelid) + str(ipid)
            d[k]['list'].append(ut.div0(d[k]['cnt'][cipid],
                                        d[ccc]['cnt'][channelid]))
        elif k == 'app_ip_click_ratio':
            aipid = str(appid) + str(ipid)
            d[k]['list'].append(ut.div0(d[k]['cnt'][aipid],
                                        d[acc]['cnt'][appid]))
        elif k == 'channel_ip_hour_ratio':
            ciphid = '%d%d%d%d' % (dayid, hourid, channelid, ipid)
            iphid = '%d%d%d' % (dayid, hourid, ipid)
            d[k]['list'].append(ut.div0(d[k]['cnt'][ciphid],
                                        d[k]['cnt'][iphid]))
        elif k == 'app_ip_hour_ratio':
            aiphid = '%d%d%d%d' % (dayid, hourid, appid, ipid)
            iphid = '%d%d%d' % (dayid, hourid, ipid)
            d[k]['list'].append(ut.div0(d[k]['cnt'][aiphid],
                                        d[k]['cnt'][iphid]))

    def _update_dict(self, d, k, v):
        uid = v.get('user', None)
        text = v.get('text', None)
        ipid = v.get('ip', None)
        appid = v.get('app', None)
        channelid = v.get('channel', None)
        dayid = v.get('day', None)
        hourid = v.get('hour', None)
        minuteid = v.get('minute', None)

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
        elif k == 'ip_click_count':
            d[k]['cnt'][ipid] += 1
        elif k == 'app_click_count':
            d[k]['cnt'][appid] += 1
        elif k == 'channel_click_count':
            d[k]['cnt'][channelid] += 1
        elif k == 'channel_ip_click_ratio':
            d[k]['cnt'][str(channelid) + str(ipid)] += 1
        elif k == 'app_ip_click_ratio':
            d[k]['cnt'][str(appid) + str(ipid)] += 1
        elif k == 'channel_ip_hour_ratio':
            ciphid = '%d%d%d%d' % (dayid, hourid, channelid, ipid)
            iphid = '%d%d%d' % (dayid, hourid, ipid)
            d[k]['cnt'][ciphid] += 1
            d[k]['cnt'][iphid] += 1
        elif k == 'app_ip_hour_ratio':
            aiphid = '%d%d%d%d' % (dayid, hourid, appid, ipid)
            iphid = '%d%d%d' % (dayid, hourid, ipid)
            d[k]['cnt'][aiphid] += 1
            d[k]['cnt'][iphid] += 1
