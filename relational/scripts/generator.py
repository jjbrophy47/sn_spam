"""
Module to generate ids for relationships between data points.
"""
import os
import re
import pandas as pd


class Generator:
    """Class that handles operations to group comments of a certain relation
    together."""

    # public
    def gen_group_ids(self, df, relations):
        """Generates missing relationsfor a given dataframe.
        df: comments dataframe.
        relations: list of tuples each specifying a different relation.
        Returns dataframe with filled in relations."""
        for relation, group, group_id in relations:
            print('\t' + relation + '...')
            df = self.gen_group_id(df, group_id)
        return df

    def gen_rel_df(self, df, group_id, data_dir=None):
        """Creates identifiers to group comments with if missing.
        df: comments dataframe.
        group_id: column identifier to group comments by.
        Returns df with the specified identifier added."""
        df = df.copy()

        if group_id not in list(df):
            if group_id == 'text_id':
                r_df = self.gen_text_ids(df, group_id, data_dir)
            elif group_id == 'hash_id':
                r_df = self.gen_string_ids(df, group_id, regex=r'(#\w+)')
            elif group_id == 'ment_id':
                r_df = self.gen_string_ids(df, group_id, regex=r'(@\w+)')
            elif group_id == 'link_id':
                r_df = self.gen_string_ids(df, group_id, regex=r'(http[^\s]+)',
                        data_dir=data_dir)
            elif group_id == 'hour_id':
                r_df = self.gen_hour_ids(df, group_id)
        else:
            r_df = df
        return r_df

    # private
    def gen_text_ids(self, df, g_id, data_dir=None):
        """Generates text ids for comments that match each other.
        df: dataframe of comments, must contain column 'text'.
        g_id: group identifier.
        Returns dataframe with text ids."""
        if data_dir is not None and os.path.exists(data_dir + 'msg_sim.csv'):
            r_df = pd.read_csv(data_dir + 'msg_sim.csv')
            r_df = r_df[r_df['com_id'].isin(df['com_id'])]
            g_df = r_df.groupby(g_id).size().reset_index()
            g_df = g_df[g_df[0] > 1]
            r_df = r_df[r_df[g_id].isin(g_df[g_id])]
        else:
            df['text'] = df['text'].fillna('')
            g_df = df.groupby('text').size().reset_index()
            g_df.columns = ['text', 'size']
            g_df = g_df[g_df['size'] > 1]
            g_df[g_id] = list(range(1, len(g_df) + 1))
            g_df = g_df.drop(['size'], axis=1)
            r_df = df.merge(g_df, on='text')
            r_df = r_df.filter(items=['com_id', g_id])
            r_df[g_id] = r_df[g_id].apply(int)
        return r_df

    def gen_hour_ids(self, df, g_id):
        """Extracts the hour from the comment timestamp.
        df: dataframe of comments, must contain 'timestamp' column with the
                form: '2011-10-31 13:37:50'.
        g_id: group identifier.
        Returns dataframe with hour_id numbers."""
        r_df = df.copy()
        r_df[g_id] = r_df['timestamp'].astype(str).str[11:13]
        r_df = r_df.filter(items=['com_id', g_id])
        return r_df

    def gen_string_ids(self, df, g_id, regex=r'(#\w+)', data_dir=None):
        """Extracts item ids (e.g. #hashtags, @mentions) from the comments.
        df: dataframe of coments.
        g_id: group identifier.
        Returns dataframe with ids as a string for each com_id."""
        if data_dir is not None and os.path.exists(data_dir + 'link_sim.csv'):
            r_df = pd.read_csv(data_dir + 'link_sim.csv')
            r_df = r_df[r_df['com_id'].isin(df['com_id'])]
            g_df = r_df.groupby(g_id).size().reset_index()
            g_df = g_df[g_df[0] > 1]
            r_df = r_df[r_df[g_id].isin(g_df[g_id])]

        else:
            regex = re.compile(regex)
            inrel = []

            for _, row in df.iterrows():
                s = self.get_items(row.text, regex)
                inrel.append({'com_id': row.com_id, g_id: s})

            inrel_df = pd.DataFrame(inrel).drop_duplicates()
            inrel_df = inrel_df.query(g_id + ' != ""')
            r_df = df.merge(inrel_df, on='com_id')
            r_df = r_df.filter(items=['com_id', g_id])

        return r_df

    def get_items(self, text, regex, str_form=True):
        """Method to extract hashtags from a string of text.
        text: text of the comment.
        regex: regex to extract items from the comment.
        str_form: concatenates list of items if True.
        Returns a string or list of item ids."""
        items = regex.findall(text)[:10]
        result = sorted([x.lower() for x in items])
        if str_form:
            result = ''.join(result)
        return result
