"""
This module takes a data set with labels and relations, then
tests how well each relation separates spam and ham into groups.
"""
import pandas as pd


class Purity:
    """Handles all operations to test purity of a data set."""

    def __init__(self, config_obj, generator_obj, util_obj):
        """Initialize dependencies."""

        self.config_obj = config_obj
        """User settings."""
        self.generator_obj = generator_obj
        """Finds and generates relational ids."""
        self.util_obj = util_obj
        """General utility methods."""

    # public
    def test_relations(self, df):
        """Checks the mixture of labels for each relational group. A good
                relational group will have almost all spam or ham.
        df: comments dataframe."""
        s = 'Condition #1: How well does each relation separate spam/ham...'
        s += '\nScale is from 0.0 to 0.5, good to bad:'
        print(s)
        relations = self.config_obj.relations

        data_f, _ = self.define_file_folders()
        df = self.read_comments(df, data_f)
        filled_df = self.gen_group_ids(df)
        self.check_relations(filled_df, relations)

    # private
    def define_file_folders(self):
        """Returns absolute path directories."""
        ind_dir = self.config_obj.ind_dir
        domain = self.config_obj.domain

        data_f = ind_dir + 'data/' + domain + '/'
        pred_f = ind_dir + 'output/' + domain + '/predictions/'
        return data_f, pred_f

    def read_comments(self, df, data_f):
        """Reads the comments if the dataframe is empty.
        df: comments dataframe.
        data_f: data folder.
        Returns the coments dataframe."""
        start = self.config_obj.start
        end = self.config_obj.end
        modified = self.config_obj.modified

        if df is None:
            name = self.util_obj.get_comments_filename(modified)
            filename = data_f + name

            if self.util_obj.check_file(filename):
                df = pd.read_csv(filename, nrows=end)
                df = df[start:]
                df = df.reset_index()
                df = df.drop(['index'], axis=1)
        return df

    def gen_group_ids(self, df):
        """Generates any missing group_id columns.
        df: comments dataframe with predictions.
        Returns dataframe with filled in group_ids."""
        for relation, group, group_id in self.config_obj.relations:
            df = self.generator_obj.gen_group_id(df, group_id)
        return df

    def check_relations(self, df, relations):
        """Checks how well each relation separates spam and ham comments.
        df: comments dataframe.
        relations: list of tuples describing each relation."""
        for relation, group, group_id in relations:
            temp_df = df[~df[group_id].isin(['empty'])]
            g_df = temp_df.groupby(group_id).size().reset_index()
            g_df.columns = [group_id, 'size']
            g_df = g_df.query('size > 1')
            num_coms = g_df['size'].sum()
            rel_score = self.check_groups(df, group_id, list(g_df[group_id]))
            print(relation + ': %.3f' % (rel_score / num_coms))

    def check_groups(self, df, group_id, group_id_vals):
        """Keeps a running score for all groups in a relation.
        df: comments dataframe.
        group_id: identifier for the relation.
        group_id_vals: group identifiers for the relation.
        Returns overall score comprised of all groups in the relation."""
        score = 0

        for group_id_val in group_id_vals:
            g_df = df[df[group_id] == group_id_val]
            group_score = self.check_group(g_df)
            score += group_score
        return score

    def check_group(self, g_df):
        """Checks to see how many spam and ham are mixed in one group.
        g_df: group dataframe.
        Returns higher score for highly mixed (max: half spam, half ham), or a
                lower score for less mixed (min: all spam or all ham). This
                score is weighted by the number of comments in the group."""
        num_coms = len(g_df)
        num_spam = g_df['label'].sum()
        num_ham = num_coms - num_spam

        mean = g_df['label'].mean()
        if num_spam > num_ham:
            mean = 1.0 - mean
        score = num_coms * mean
        return score
