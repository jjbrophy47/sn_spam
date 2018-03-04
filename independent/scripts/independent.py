"""
Module containing the Independent class to handle all operations pertaining
to the independent model.
"""
import os
import numpy as np
import pandas as pd


class Independent:
    """Returns an Independent object that reads in the data, splits into sets,
    trains and classifies, and writes the results."""

    def __init__(self, config_obj, classification_obj, generator_obj,
            util_obj):
        self.config_obj = config_obj
        self.classification_obj = classification_obj
        self.gen_obj = generator_obj
        self.util_obj = util_obj

    # public
    def main(self, coms_df, stacking=0, alter_user_ids=False, separate_relations=False,
            separate_models=False, modified=False):
        """Main method that reads in the comments, splits them into train and
        test, writes them to files, and prints out stats.
        Returns the train and test comment dataframes."""
        self.util_obj.start()
        data_f, fold_f, status_f = self.file_folders()
        sw = self.open_status_writer(status_f)

        # coms_filename = self.util_obj.get_comments_filename(modified)
        # coms_df = self.read_file(data_f + coms_filename, sw)
        train_df, val_df, test_df = self.split_coms(coms_df)

        if alter_user_ids:
            test_df = self.alter_user_ids(coms_df, test_df)

        if separate_relations:
            val_df = self.separate_relations(coms_df, train_df, val_df)
            test_df = self.separate_relations(coms_df, train_df, test_df)

        self.write_folds(val_df, test_df, fold_f)
        self.print_subsets(train_df, val_df, test_df, fw=sw)

        self.util_obj.start('\nvalidation set:\n', fw=sw)
        self.classification_obj.main(train_df, val_df, dset='val', fw=sw)
        self.util_obj.end('time: ', fw=sw)

        self.util_obj.start('\ntest set:\n', fw=sw)
        all_train_df = train_df.copy()
        if self.config_obj.super_train:
            all_train_df = pd.concat([train_df, val_df])

        self.classification_obj.main(all_train_df, test_df, dset='test', fw=sw)
        self.util_obj.end('time: ', fw=sw)

        self.util_obj.end('total independent model time: ', fw=sw)
        self.util_obj.close_writer(sw)
        return val_df, test_df

    # private
    def file_folders(self):
        """Returns absolute paths for various directories."""
        ind_dir = self.config_obj.ind_dir
        domain = self.config_obj.domain

        data_f = ind_dir + 'data/' + domain + '/'
        fold_f = ind_dir + 'data/' + domain + '/folds/'
        status_f = ind_dir + 'output/' + domain + '/status/'
        if not os.path.exists(fold_f):
            os.makedirs(fold_f)
        if not os.path.exists(status_f):
            os.makedirs(status_f)
        return data_f, fold_f, status_f

    def open_status_writer(self, status_f):
        """Opens a file to write updates of the independent model.
        status_f: status folder.
        Returns file object to write to."""
        fold = self.config_obj.fold
        fname = status_f + 'ind_' + fold + '.txt'
        f = self.util_obj.open_writer(fname)
        return f

    def read_file(self, filename, fw=None):
        """Reads the appropriate comments file of the domain.
        filename: csv comments file.
        Returns comments dataframe up to the end marker in the config."""
        self.util_obj.start('loading data...', fw=fw)
        coms_df = pd.read_csv(filename, lineterminator='\n',
                              nrows=self.config_obj.end)
        self.util_obj.end(fw=fw)
        return coms_df

    def split_coms(self, coms_df):
        """Splits the comments into training, validation, and test sets.
        coms_df: comments dataframe.
        Returns train, val, and test dataframes."""
        start = self.config_obj.start
        train_size = self.config_obj.train_size
        val_size = self.config_obj.val_size

        coms_df = coms_df[start:]
        num_coms = len(coms_df)
        split_ndx1 = int(num_coms * train_size)
        split_ndx2 = split_ndx1 + int(num_coms * val_size)

        train_df = coms_df[:split_ndx1]
        val_df = coms_df[split_ndx1:split_ndx2]
        test_df = coms_df[split_ndx2:]

        return train_df, val_df, test_df

    def alter_user_ids(self, coms_df, test_df):
        """Alters the user ids in the test set so that all spam messages
                are posted by a different user.
        test_df: test set dataframe.
        Returns altered test set with different user ids for each spammer."""
        max_user_id = coms_df['user_id'].max() + 1
        user_ids = list(zip(test_df['label'], test_df['user_id']))
        new_user_ids = []

        for label, user_id in user_ids:
            new_user_ids.append(max_user_id if label == 1 else user_id)
            max_user_id += 1
        test_df['user_id'] = new_user_ids
        return test_df

    def separate_relations(self, coms_df, train_df, test_df):
        for relation, group, g_id in self.config_obj.relations:

            if relation in ['posts', 'intrack', 'invideo']:
                nc = 'n' + g_id
                max_id = test_df[g_id].max() + 1

                df = test_df[test_df[g_id].isin(train_df[g_id])]
                g_df = df.groupby(g_id).size().reset_index()
                g_df[nc] = range(max_id, max_id + len(g_df))
                g_df = g_df.drop([0], axis=1)

                test_df = test_df.merge(g_df, on=g_id, how='left')
                choose = lambda r: r[g_id] if np.isnan(r[nc]) else r[nc]
                test_df[g_id] = test_df.apply(choose, axis=1)
                test_df = test_df.drop([nc], axis=1)
                test_df[g_id] = test_df[g_id].apply(int)

            elif relation in ['intext', 'inment', 'inhash', 'inlink']:
                c, nc = 'text', 'n_text'
                c_df = self.gen_obj.gen_rel_df(coms_df, g_id)
                tr_df = train_df.merge(c_df, on='com_id')
                te_df = test_df.merge(c_df, on='com_id')
                df = te_df[te_df[g_id].isin(tr_df[g_id])]

                if relation == 'intext':
                    df[nc] = df[c] + '.'
                if relation == 'inment':
                    df[nc] = df[c].str.replace('@', '@-')
                if relation == 'inhash':
                    df[nc] = df[c].str.replace('#', '#-')
                if relation == 'inlink':
                    df[nc] = df[c].str.replace('https', 'https-')

                df = df[['com_id', nc]]
                test_df = test_df.merge(df, on='com_id', how='left')
                choose = lambda r: r[c] if pd.isnull(r[nc]) else r[nc]
                test_df[c] = test_df.apply(choose, axis=1)
                test_df = test_df.drop([nc], axis=1)

            return test_df

    def write_folds(self, val_df, test_df, fold_f):
        """Writes validation and test set dataframes to csv files.
        val_df: dataframe with validation set comments.
        test_df: dataframe with test set comments.
        fold_f: folder to save the data to."""
        fold = self.config_obj.fold
        val_fname = fold_f + 'val_' + fold + '.csv'
        test_fname = fold_f + 'test_' + fold + '.csv'

        val_df.to_csv(val_fname, line_terminator='\n', index=None)
        test_df.to_csv(test_fname, line_terminator='\n', index=None)

    def print_subsets(self, train_df, val_df, test_df, fw=None):
        """Writes basic statistics about the training and test sets.
        train_df: training set comments.
        test_df: test set comments."""
        spam, total = len(train_df[train_df['label'] == 1]), len(train_df)
        percentage = round(self.util_obj.div0(spam, total) * 100, 1)
        s = '\ttraining set size: ' + str(len(train_df)) + ', '
        s += 'spam: ' + str(spam) + ' (' + str(percentage) + '%)'
        self.util_obj.write(s, fw=fw)

        spam, total = len(val_df[val_df['label'] == 1]), len(val_df)
        percentage = round(self.util_obj.div0(spam, total) * 100, 1)
        s = '\tvalidation set size: ' + str(len(val_df)) + ', '
        s += 'spam: ' + str(spam) + ' (' + str(percentage) + '%)'
        self.util_obj.write(s, fw=fw)

        spam, total = len(test_df[test_df['label'] == 1]), len(test_df)
        percentage = round(self.util_obj.div0(spam, total) * 100, 1)
        s = '\ttest set size: ' + str(len(test_df)) + ', '
        s += 'spam: ' + str(spam) + ' (' + str(percentage) + '%)'
        self.util_obj.write(s, fw=fw)
