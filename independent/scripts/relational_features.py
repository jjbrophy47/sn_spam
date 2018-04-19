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
    def build(self, df, dset, stack=0):
        t1 = self.util_obj.out('building relational features...')

        feats_df, feats_list = self._build_features(df, stack=stack)
        feats_list = [x for x in feats_list if x != 'com_id']

        self.util_obj.time(t1)
        return feats_df, feats_list

    # private
    def _build_features(self, df, stack=0):
        featuresets = self.config_obj.featuresets
        rels = [r[0] for r in self.config_obj.relations]
        usr = 'user' if 'posts' in rels else 'user_id'
        fdf = df.copy()

        if self.config_obj.domain == 'adclicks':
            fl = ['com_id', 'ip_cnt', 'app_cnt', 'chn_cnt', 'chn_ip_cnt',
                  'chn_ip_rto', 'app_ip_cnt', 'app_ip_rto']

            # 'chn_ip_hour_ratio', 'app_ip_hour_ratio']

            fdf['ip_cnt'] = fdf.groupby('ip').cumcount()
            fdf['app_cnt'] = fdf.groupby('app').cumcount()
            fdf['chn_cnt'] = fdf.groupby('channel').cumcount()
            fdf['chn_ip_cnt'] = fdf.groupby(['channel', 'ip']).cumcount()
            fdf['app_ip_cnt'] = fdf.groupby(['app', 'ip']).cumcount()
            fdf['chn_ip_rto'] = fdf.chn_ip_cnt.divide(fdf.chn_cnt).fillna(0)
            fdf['app_ip_rto'] = fdf.app_ip_cnt.divide(fdf.app_cnt).fillna(0)

        elif self.config_obj.domain == 'soundcloud':
            fl = ['com_id', 'usr_msg_cnt', 'usr_lnk_rto']

            fdf['has_lnk'] = fdf.text.apply(lambda x: 1 if 'http' in x else 0)
            lnk_cnt = fdf.groupby(usr).has_lnk.cumsum() - fdf.has_lnk

            fdf['usr_msg_cnt'] = fdf.groupby(usr).cumcount()
            fdf['usr_lnk_rto'] = lnk_cnt.divide(fdf.usr_msg_cnt).fillna(0)

            fdf = fdf[fl]

        elif self.config_obj.domain == 'youtube':
            fl = ['com_id', 'usr_msg_cnt', 'usr_msg_max', 'usr_msg_min',
                  'usr_msg_mean']

            fdf['len'] = fdf.text.str.len()

            fdf['usr_msg_cnt'] = fdf.groupby(usr).cumcount()
            fdf['usr_msg_max'] = fdf.groupby(usr)['len'].cummax()
            fdf['usr_msg_min'] = fdf.groupby(usr)['len'].cummin()
            fdf['usr_msg_mean'] = list(fdf.groupby(usr)['len']
                                       .expanding().mean().reset_index()
                                       .sort_values('level_1')['len'])

            fdf = fdf[fl]

        elif self.config_obj.domain == 'twitter':
            fl = ['com_id', 'usr_msg_cnt', 'usr_lnk_rto', 'usr_hsh_rto',
                  'usr_men_rto']

            fdf['has_lnk'] = fdf.text.apply(lambda x: 1 if 'http' in x else 0)
            fdf['has_hsh'] = fdf.text.apply(lambda x: 1 if '#' in x else 0)
            fdf['has_men'] = fdf.text.apply(lambda x: 1 if '@' in x else 0)
            lnk_cnt = fdf.groupby(usr)['has_lnk'].cumsum() - fdf.has_lnk
            hsh_cnt = fdf.groupby(usr).has_hsh.cumsum() - fdf.has_hsh
            men_cnt = fdf.groupby(usr).has_men.cumsum() - fdf.has_men

            fdf['usr_msg_cnt'] = fdf.groupby(usr).cumcount()
            fdf['usr_lnk_rto'] = lnk_cnt.divide(fdf.usr_msg_cnt).fillna(0)
            fdf['usr_hsh_rto'] = hsh_cnt.divide(fdf.usr_msg_cnt).fillna(0)
            fdf['usr_men_rto'] = men_cnt.divide(fdf.usr_msg_cnt).fillna(0)

            fdf = fdf[fl]

        elif self.config_obj.domain == 'russia':
            fl = ['com_id', 'usr_msg_cnt', 'usr_lnk_rto', 'usr_hsh_rto',
                  'usr_men_rto']

        elif self.config_obj.domain == 'toxic':
            fl = ['com_id']

        if not any(x in featuresets for x in ['sequential', 'all']):
            fdf, fl = pd.DataFrame(df['com_id']), []

        if stack > 0:  # add pseduo-relational features
            fdf2, fl2 = self._build_pseudo_relational_features(df)
            fdf = fdf.merge(fdf2, on='com_id', how='left')
            fl += fl2

        fl.remove('com_id')
        fdf = fdf.drop(['app', 'ip', 'os', 'channel', 'device', 'click_time',
                        'attributed_time', 'label'], axis=1)
        return fdf, fl

    def _build_pseudo_relational_features(self, df):
        h, d = self._init_headers_and_super_dict(df)

        for r in df.itertuples():
            com_id, noisy_label = r[h['com_id']], r[h['noisy_label']]
            d['com_id'].append(com_id)
            self._update_relational(d, r, h, noisy_label)

        feats_df, feats_list = self._build_dataframe(d)
        return feats_df, feats_list

    def _init_headers_and_super_dict(self, df):
        domain = self.config_obj.domain
        label_name = 'spam' if domain != 'adclicks' else 'attribution'

        headers = list(df)
        h = {h: i + 1 for i, h in enumerate(headers)}

        d = {'com_id': []}
        for relation, group, group_id in self.config_obj.relations:
            key = group + '_' + label_name + '_rto'
            d[key] = {'label': defaultdict(float), 'cnt': defaultdict(int),
                      'list': []}
        return h, d

    def _update_relational(self, d, row, headers, noisy_label):
        ut = self.util_obj
        domain = self.config_obj.domain
        label_name = 'spam' if domain != 'adclicks' else 'attribution'

        for relation, group, group_id in self.config_obj.relations:
            rd = d[group + '_' + label_name + '_rto']
            rel_ids = row[headers[group_id]]

            ratios = []
            for rel_id in rel_ids:
                ratios.append(ut.div0(rd['label'][rel_id], rd['cnt'][rel_id]))
                rd['cnt'][rel_id] += 1
                rd['label'][rel_id] += noisy_label

            rto_mean = np.mean(ratios)
            rd['list'].append(0 if np.isnan(rto_mean) else rto_mean)

    def _build_dataframe(self, d):
        cols = ['com_id']
        lists = [d['com_id']]

        for k, v in d.items():
            if type(v) == dict:
                cols.append(k)
                lists.append(v['list'])

        feats = list(zip(*lists))
        feats_df = pd.DataFrame(feats, columns=cols)
        feats_list = list(feats_df)

        return feats_df, feats_list
