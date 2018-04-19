"""
This module creates relational features in sequential order of messages.
"""
import numpy as np
import pandas as pd
import scipy.sparse as ss
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer


class Features:

    def __init__(self, config_obj, util_obj):
        self.config_obj = config_obj
        self.util_obj = util_obj

    # public
    def build(self, df, dset, stack=0, cv=None):

        featuresets = self.config_obj.featuresets
        rels = [r[0] for r in self.config_obj.relations]
        usr = 'user' if 'posts' in rels else 'user_id'

        fdf, fl, m = df.copy(), [], None

        if self.config_obj.domain == 'adclicks':

            if any(x in featuresets for x in ['content', 'all']):
                t1 = self.util_obj.out('building content features...')
                in_h = lambda x: 1 if x in [4, 5, 9, 10, 13, 14] else \
                    2 if x in [6, 11, 15] else 3
                fdf['click_time'] = pd.to_datetime(fdf['click_time'])

                fdf['wday'] = fdf['click_time'].dt.dayofweek
                fdf['hour'] = fdf['click_time'].dt.hour
                fdf['min'] = fdf['click_time'].dt.minute
                fdf['usr_cnt'] = self._count(['ip', 'device', 'os'], fdf)
                fdf['usr_app_cnt'] = self._count(['ip', 'device', 'os',
                                                  'app'], fdf)
                fdf['in_h'] = fdf['hour'].apply(in_h)
                fdf['n_app'] = self._count(['app', 'wday', 'hour'], fdf)
                fdf['n_ip'] = self._count(['ip', 'wday', 'hour'], fdf)
                fdf['n_ip_app'] = self._count(['ip', 'wday', 'hour', 'app'],
                                              fdf)
                fdf['n_ip_os'] = self._count(['ip', 'wday', 'hour', 'os'], fdf)
                fdf['n_ip_app_os'] = self._count(['ip', 'wday', 'hour', 'app',
                                                  'os'], fdf)
                fdf['nip_day_h'] = self._count(['ip', 'wday', 'in_h'], fdf)
                fl += ['wday', 'hour', 'min', 'usr_cnt', 'usr_app_cnt',
                       'n_app', 'n_ip', 'n_ip_app', 'n_ip_os', 'n_ip_app_os',
                       'nip_day_h']

                self.util_obj.time(t1)

            if any(x in featuresets for x in ['sequential', 'all']):
                t1 = self.util_obj.out('building sequential features...')

                fdf['usr_new'] = fdf.groupby(['ip', 'device', 'os']).cumcount()
                fdf['usr_app_new'] = fdf.groupby(['ip', 'device', 'os',
                                                  'app']).cumcount()
                fdf['ip_cnt'] = fdf.groupby('ip').cumcount()
                fdf['app_cnt'] = fdf.groupby('app').cumcount()
                fdf['chn_cnt'] = fdf.groupby('channel').cumcount()
                fdf['chn_ip_cnt'] = fdf.groupby(['channel', 'ip']).cumcount()
                fdf['app_ip_cnt'] = fdf.groupby(['app', 'ip']).cumcount()
                fdf['chn_ip_rto'] = fdf.chn_ip_cnt.divide(fdf.chn_cnt)\
                                       .fillna(0)
                fdf['app_ip_rto'] = fdf.app_ip_cnt.divide(fdf.app_cnt)\
                                       .fillna(0)
                fl += ['usr_new', 'usr_app_new', 'ip_cnt', 'app_cnt',
                       'chn_cnt', 'chn_ip_cnt', 'app_ip_cnt', 'chn_ip_rto',
                       'app_ip_rto']

                self.util_obj.time(t1)

        elif self.config_obj.domain == 'soundcloud':

            if any(x in featuresets for x in ['ngrams', 'all']):
                m, cv = self._ngrams(fdf, cv=cv)

            if any(x in featuresets for x in ['content', 'all']):
                t1 = self.util_obj.out('building content features...')

                fdf['num_chs'] = fdf.text.str.len()
                fdf['has_lnk'] = fdf.text.str.contains('http').astype(int)
                fl += ['num_chs', 'has_lnk', 'polarity', 'subjectivity']

                self.util_obj.time(t1)

            if any(x in featuresets for x in ['sequential', 'all']):
                t1 = self.util_obj.out('building sequential features...')
                fdf['has_lnk'] = fdf['text'].str.contains('http') \
                                                .astype(int)
                lnk_cnt = fdf.groupby(usr).has_lnk.cumsum() - fdf.has_lnk

                l = ['usr_msg_cnt', 'usr_lnk_rto']
                fdf['usr_msg_cnt'] = fdf.groupby(usr).cumcount()
                fdf['usr_lnk_rto'] = lnk_cnt.divide(fdf.usr_msg_cnt).fillna(0)
                fl += ['usr_msg_cnt', 'usr_lnk_rto']

                self.util_obj.time(t1)

        elif self.config_obj.domain == 'youtube':

            if any(x in featuresets for x in ['ngrams', 'all']):
                m, cv = self._ngrams(fdf, cv=cv)

            if any(x in featuresets for x in ['content', 'all']):
                t1 = self.util_obj.out('building content features...')
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                fdf['num_chs'] = df['text'].str.len()
                fdf['wday'] = df['timestamp'].dt.dayofweek
                fdf['hour'] = df['timestamp'].dt.hour
                fl += ['num_chs', 'wday', 'hour', 'polarity', 'subjectivity']

                self.util_obj.time(t1)

            if any(x in featuresets for x in ['sequential', 'all']):
                t1 = self.util_obj.out('building sequential features...')
                fdf['len'] = fdf.text.str.len()

                fdf['usr_msg_cnt'] = fdf.groupby(usr).cumcount()
                fdf['usr_msg_max'] = fdf.groupby(usr)['len'].cummax()
                fdf['usr_msg_min'] = fdf.groupby(usr)['len'].cummin()
                fdf['usr_msg_mean'] = list(fdf.groupby(usr)['len']
                                           .expanding().mean().reset_index()
                                           .sort_values('level_1')['len'])
                fl += ['com_id', 'usr_msg_cnt', 'usr_msg_max', 'usr_msg_min',
                       'usr_msg_mean']

                self.util_obj.time(t1)

        elif self.config_obj.domain == 'twitter':

            if any(x in featuresets for x in ['graph', 'all']):
                t1 = self.util_obj.out('building graph features...')
                fl += ['pagerank', 'triangle_count', 'core_id', 'out_degree',
                       'in_degree']
                self.util_obj.time(t1)

            if any(x in featuresets for x in ['ngrams', 'all']):
                m, cv = self._ngrams(fdf, cv=cv)

            if any(x in featuresets for x in ['content', 'all']):
                t1 = self.util_obj.out('building content features...')

                fdf['num_chs'] = df['text'].str.len()
                fdf['num_hsh'] = df['text'].str.count('#')
                fdf['num_men'] = df['text'].str.count('@')
                fdf['num_lnk'] = df['text'].str.count('http')
                fdf['num_rtw'] = df['text'].str.count('RT')
                fl += ['num_chs', 'num_hsh', 'num_men', 'num_lnk', 'num_rtw']

                self.util_obj.time(t1)

            if any(x in featuresets for x in ['sequential', 'all']):
                t1 = self.util_obj.out('building sequential features...')

                fdf['has_lnk'] = fdf.text.str.contains('http').astype(int)
                fdf['has_hsh'] = fdf.text.str.contains('#').astype(int)
                fdf['has_men'] = fdf.text.str.contains('@').astype(int)
                lnk_cnt = fdf.groupby(usr)['has_lnk'].cumsum() - fdf.has_lnk
                hsh_cnt = fdf.groupby(usr).has_hsh.cumsum() - fdf.has_hsh
                men_cnt = fdf.groupby(usr).has_men.cumsum() - fdf.has_men

                fdf['usr_msg_cnt'] = fdf.groupby(usr).cumcount()
                fdf['usr_lnk_rto'] = lnk_cnt.divide(fdf.usr_msg_cnt).fillna(0)
                fdf['usr_hsh_rto'] = hsh_cnt.divide(fdf.usr_msg_cnt).fillna(0)
                fdf['usr_men_rto'] = men_cnt.divide(fdf.usr_msg_cnt).fillna(0)
                fl += ['usr_msg_cnt', 'usr_lnk_rto', 'usr_hsh_rto',
                       'usr_men_rto']

                self.util_obj.time(t1)

            fdf = fdf[fl]

        if any(x in featuresets for x in ['pseudo', 'all']) and stack > 0:
            t1 = self.util_obj.out('building pseudo-relational features...')
            prf, l = self._build_pseudo_relational_features(df)

            fdf = fdf.reset_index().drop(['index'], axis=1)
            fdf = fdf.join(prf, how='left')
            fl += l
            self.util_obj.time(t1)

        fdf = fdf[fl]
        return fdf, fl, m, cv

    # private
    def _build_pseudo_relational_features(self, df):
        h, d = self._init_headers_and_super_dict(df)

        for r in df.itertuples():
            noisy_label = r[h['noisy_label']]
            self._update_relational(d, r, h, noisy_label)

        feats_df, feats_list = self._build_dataframe(d)
        return feats_df, feats_list

    def _init_headers_and_super_dict(self, df):
        domain = self.config_obj.domain
        label_name = 'spam' if domain != 'adclicks' else 'attribution'

        headers = list(df)
        h = {h: i + 1 for i, h in enumerate(headers)}

        d = {}
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
        cols = []
        lists = []

        for k, v in d.items():
            if type(v) == dict:
                cols.append(k)
                lists.append(v['list'])

        feats = list(zip(*lists))
        feats_df = pd.DataFrame(feats, columns=cols)
        feats_list = list(feats_df)

        return feats_df, feats_list

    def _ngrams(self, fdf, cv=None):
        t1 = self.util_obj.out('building ngrams...')

        fdf.text = fdf.text.fillna('')
        str_list = fdf.text.tolist()

        if cv is None:
            cv = self._count_vectorizer()
            ngrams_m = cv.fit_transform(str_list)
        else:
            ngrams_m = cv.transform(str_list)

        id_m = ss.lil_matrix((len(fdf), 1))
        m = ss.hstack([id_m, ngrams_m]).tocsr()

        self.util_obj.time(t1)
        return m, cv

    def _count_vectorizer(self):
        cv = CountVectorizer(stop_words='english', min_df=1,
                             ngram_range=(3, 3), max_df=1.0,
                             max_features=10000, analyzer='char_wb',
                             binary=True, vocabulary=None, dtype=np.int32)
        return cv

    def _count(self, cols, df):
        qf1 = df.groupby(cols).size().reset_index().rename(columns={0: 'size'})
        qf2 = df.merge(qf1, how='left')
        return list(qf2['size'])
