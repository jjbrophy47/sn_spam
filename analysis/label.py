"""
This module relabels data in various ways.
"""
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

        labels_dict = self.relabel_relations(df, relations)

        if len(labels_dict) > 0:
            new_df, labels_df = self.merge_labels(df, labels_dict)
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
        d = {}

        for relation, group, group_id in relations:
            temp_df = df[~df[group_id].isin(['empty'])]
            g_df = temp_df.groupby(group_id).size().reset_index()
            g_df.columns = [group_id, 'size']
            g_df = g_df.query('size > 1')
            rel_dict = self.relabel_groups(df, group_id, list(g_df[group_id]))
            d.update(rel_dict)
        self.util_obj.end()
        return d

    def relabel_groups(self, df, group_id, group_id_vals):
        """Relabels each group of a specific relation.
        df: comments dataframe.
        group_id: identifier of the relation.
        group_id_vals: id values for each group in the relation.
        Returns dict of com ids and their labels for all groups."""
        d = {}

        for group_id_val in group_id_vals:
            g_df = df[df[group_id] == group_id_val]
            group_dict = self.relabel_group(g_df)
            d.update(group_dict)
        return d

    def relabel_group(self, g_df):
        """Relabels data points within a group.
        g_df: dataframe of comments for a related group.
        Returns dict of com ids with new labels based on majority label."""
        if self.config_obj.debug:
            print(g_df)

        num_spam = g_df['label'].sum()
        new_label = 1 if num_spam >= 1 else 0

        if self.config_obj.debug:
            print(len(g_df), g_df['label'].sum())

        alter = lambda x: -1 if x['label'] == new_label else x['com_id']
        com_ids_list = g_df.apply(alter, axis=1)
        altered_list = [(x, new_label) for x in com_ids_list if x != -1]
        d = dict(altered_list)
        return d

    def merge_labels(self, df, d):
        """Takes the dict of com ids and labels and swaps them into the
                original comments dataframe.
        df: comments dataframe.
        d: dict of com ids and new labels.
        Returns comments dataframe with new labels, dataframe with only
                the com ids whoe labels were changed."""
        new_df = df.copy()
        labels_df = pd.DataFrame.from_dict(d, orient='index').reset_index()
        labels_df.columns = ['com_id', 'new_label']
        temp_df = df.merge(labels_df, on='com_id', how='left')
        temp_df['new_label'] = temp_df['new_label'].fillna(temp_df['label'])
        new_df['label'] = temp_df['new_label'].apply(int)

        if self.config_obj.debug:
            print('\n\n')
            print(labels_df)
        print('comments relabeled: %d' % len(labels_df))

        return new_df, labels_df

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
