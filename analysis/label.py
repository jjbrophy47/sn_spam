"""
This module relabels data in various ways.
"""
import numpy as np
import pandas as pd


class Label:
    """Handles operations to relabel data."""

    def __init__(self, config_obj, generator_obj, util_obj):
        """Initializes the independent and relational objects."""

        self.config_obj = config_obj
        """User settings."""
        self.generator_obj = generator_obj
        """Generates group id values for relations."""
        self.util_obj = util_obj
        """General utility methods."""

    # public
    def relabel(self, df=None):
        """Relabels a dataframe of comments based on relations. A comment is
                relabeled if its label does not match the majority label
                of its relational group.
        df: comments dataframe."""
        all_relations = self.config_obj.relations

        self.util_obj.start()
        data_f = self.define_file_folders()
        df = self.read_comments(df, data_f)

        self.util_obj.start('generating group ids...\n')
        relations = self.filter_relations(all_relations)
        df = self.generator_obj.gen_group_ids(df, relations)
        self.util_obj.end('\ttime: ')

        labels_df, new_df = self.relabel_relations(df, relations)

        if len(labels_df) > 0:
            print('comments relabeled: %d' % len(labels_df))
            self.write_new_dataframe(new_df, labels_df, data_f)
        else:
            print('no comments needing relabeling...')
        self.util_obj.end('total time: ')

    # private
    def define_file_folders(self):
        """Returns absolute path to independent data folder."""
        ind_dir = self.config_obj.ind_dir
        domain = self.config_obj.domain

        data_f = ind_dir + 'data/' + domain + '/'
        return data_f

    def read_comments(self, df, data_f):
        """Reads in the data to be relabeled.
        df: comments dataframe, if None, then reads comments from config.
        data_f: data folder.
        Returns comments dataframe."""
        start = self.config_obj.start
        end = self.config_obj.end

        self.util_obj.start('reading comments...')
        if df is None:
            df = pd.read_csv(data_f + 'comments.csv', nrows=end)
            df = df[start:]
        self.util_obj.end()
        return df

    def filter_relations(self, relations):
        """Filters out all relations except posts and intext relations.
        relations: list of tuples of user specified relations.
        Returns filtered list of relations."""
        rels = [x for x in relations if x[0] == 'posts' or x[0] == 'intext']
        return rels

    def relabel_relations(self, df, relations):
        """Gathers all groups pertaining to each relation.
        df: comments dataframe.
        relations: relations to link data together with.
        Returns dict of com ids and their new labels for all relations."""
        self.util_obj.start('checking if any comments need relabeling...')
        df = df[~np.isnan(df['label'])]
        dfs = df[df['label'] == 1]

        labels_df = pd.DataFrame()
        for rel, g, g_id in relations:
            g = dfs.groupby(g_id).size().reset_index()
            g.columns = [g_id, 'size']
            q = df.merge(g, on=g_id)
            qq = q[q['label'] == 0]
            qq['relabel'] = 1
            labels_df = labels_df.append(qq)

        labels_df = labels_df.drop_duplicates()
        labels_df['label'] = labels_df['label'].apply(int)
        temp_df = labels_df[['com_id', 'relabel']]
        new_df = df.merge(temp_df, on='com_id', how='left')
        new_df['label'] = new_df['relabel'].fillna(new_df['label']).apply(int)
        del new_df['relabel']
        del labels_df['relabel']

        self.util_obj.end()
        return labels_df, new_df

    def write_new_dataframe(self, new_df, labels_df, data_f):
        """Writes the new dataframe and labels to separate files.
        new_df: comments dataframe with new labels.
        labels_df: dataframe with only the com ids that we changed.
        data_f: data folder."""
        self.util_obj.start('writing relabeled comments...')

        labels_df.to_csv(data_f + 'labels.csv', index=None)
        new_df.to_csv(data_f + 'modified.csv', encoding='utf-8',
                line_terminator='\n', index=None)
        self.util_obj.end()
