"""
Module to generate ids for relationships between data points.
"""
import os
import re
import numpy as np
import pandas as pd


class Generator:

    # public
    def gen_relational_ids(self, df, relations, data_dir=None):
        """Generates relational ids for a given dataframe.
        df: comments dataframe.
        relations: list of tuples each specifying a different relation.
        Returns dataframe with filled in relations."""
        df = df.copy()

        print('generating relational ids...')
        for relation, group, group_id in relations:
            print(relation + '...')
            df = self._gen_group_id(df, group_id, data_dir=data_dir)
        return df

    def gen_rel_df(self, df, group_id, data_dir=None):
        """Creates identifiers to group comments with if missing.
        df: comments dataframe.
        group_id: column identifier to group comments by.
        Returns df with the specified identifier added."""
        df = df.copy()

        if group_id not in list(df):
            if group_id == 'text_id':
                r_df = self._gen_text_ids(df, group_id, data_dir)
            elif group_id == 'hashtag_id':
                r_df = self._gen_string_ids(df, group_id, regex=r'(#\w+)',
                                            data_dir=data_dir)
            elif group_id == 'mention_id':
                r_df = self._gen_string_ids(df, group_id, regex=r'(@\w+)',
                                            data_dir=data_dir)
            elif group_id == 'link_id':
                r_df = self._gen_string_ids(df, group_id,
                                            regex=r'(http[^\s]+)',
                                            data_dir=data_dir)
            elif group_id == 'hour_id':
                r_df = self._gen_hour_ids(df, group_id)
        else:
            r_df = self._keep_relational_data(df, group_id)
        return r_df

    def rel_df_from_rel_ids(self, df, g_id):
        rows = []

        headers = list(df)
        h = {h: i + 1 for i, h in enumerate(headers)}

        for r in df.itertuples():
            com_id = r[h['com_id']]
            rel_ids = r[h[g_id]]

            for rel_id in rel_ids:
                rows.append((com_id, rel_id))

        rel_df = pd.DataFrame(rows, columns=['com_id', g_id])
        return rel_df

    # private
    def _gen_group_id(self, df, g_id, data_dir=None):
        r_df = self.gen_rel_df(df, g_id, data_dir=data_dir)

        if len(r_df) == 0:
            if g_id in list(df):
                df = df.rename(columns={g_id: g_id.replace('_id', '')})

            df[g_id] = np.nan
            df[g_id] = df[g_id].astype(object)
        else:
            g = r_df.groupby('com_id')

            d = {}
            for com_id, g_df in g:
                d[com_id] = [list(g_df[g_id])]
            r_df = pd.DataFrame.from_dict(d, orient='index').reset_index()
            r_df.columns = ['com_id', g_id]

            if g_id in list(df):
                df = df.rename(columns={g_id: g_id.replace('_id', '')})

            df = df.merge(r_df, on='com_id', how='left')

        for row in df.loc[df[g_id].isnull(), g_id].index:
            df.at[row, g_id] = []
        return df

    def _gen_text_ids(self, df, g_id, data_dir=None):
        if data_dir is not None and os.path.exists(data_dir + 'text_sim.csv'):
            r_df = pd.read_csv(data_dir + 'msg_sim.csv')
            r_df = r_df[r_df['com_id'].isin(df['com_id'])]
            g_df = r_df.groupby(g_id).size().reset_index()
            g_df = g_df[g_df[0] > 1]
            r_df = r_df[r_df[g_id].isin(g_df[g_id])]
        else:
            r_df = self._text_to_ids(df, g_id=g_id)
        return r_df

    def _gen_hour_ids(self, df, g_id):
        r_df = df.copy()
        r_df[g_id] = r_df['timestamp'].astype(str).str[11:13]
        r_df = r_df.filter(items=['com_id', g_id])
        return r_df

    def _gen_string_ids(self, df, g_id, regex=r'(#\w+)', data_dir=None):
        fp = ''
        if data_dir is not None:
            hash_path = data_dir + 'hashtag_sim.csv'
            ment_path = data_dir + 'mention_sim.csv'
            link_path = data_dir + 'link_sim.csv'

            if regex == r'(#\w+)':
                fp = hash_path
            elif regex == r'(@\w+)':
                fp = ment_path
            elif regex == r'(http[^\s]+)':
                fp = link_path

        if data_dir is not None and os.path.exists(fp):
            r_df = pd.read_csv(fp)
            r_df = r_df[r_df['com_id'].isin(df['com_id'])]
            g_df = r_df.groupby(g_id).size().reset_index()
            g_df = g_df[g_df[0] > 1]
            r_df = r_df[r_df[g_id].isin(g_df[g_id])]
            print(r_df)

        else:
            group = g_id.replace('_id', '')
            regex = re.compile(regex)
            inrel = []

            for _, row in df.iterrows():
                s = self._get_items(row.text, regex)
                inrel.append({'com_id': row.com_id, group: s})

            inrel_df = pd.DataFrame(inrel).drop_duplicates()
            r_df = self._text_to_ids(inrel_df, g_id=g_id)
        return r_df

    def _get_items(self, text, regex, str_form=True):
        items = regex.findall(text)[:10]
        result = sorted([x.lower() for x in items])
        if str_form:
            result = ''.join(result)
        return result

    def _keep_relational_data(self, df, g_id):
        g_df = df.groupby(g_id).size().reset_index()
        g_df = g_df[g_df[0] > 1]
        r_df = df[df[g_id].isin(g_df[g_id])]
        return r_df

    def _text_to_ids(self, df, g_id='text_id'):
        group = g_id.replace('_id', '')
        df = df[df[group] != '']

        g_df = df.groupby(group).size().reset_index()
        g_df.columns = [group, 'size']
        g_df = g_df[g_df['size'] > 1]
        g_df[g_id] = list(range(1, len(g_df) + 1))
        g_df = g_df.drop(['size'], axis=1)

        r_df = df.merge(g_df, on=group)
        r_df = r_df.filter(items=['com_id', g_id])
        r_df[g_id] = r_df[g_id].apply(int)
        return r_df
