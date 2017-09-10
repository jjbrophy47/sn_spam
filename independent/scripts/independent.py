"""
Module containing the Independent class to handle all operations pertaining
to the independent model.
"""
import os
import pandas as pd


class Independent:
    """Returns an Independent object that reads in the data, splits into sets,
    trains and classifies, and writes the results."""

    def __init__(self, config_obj, classification_obj, util_obj):
        """Initializes object dependencies for this class."""

        self.config_obj = config_obj
        """Configuration object with user settings."""
        self.classification_obj = classification_obj
        """Object that handles classification of the data."""
        self.util_obj = util_obj
        """Class containing general utility methods."""

    # private
    def define_file_folders(self):
        """Returns absolute paths for various directories."""
        ind_dir = self.config_obj.ind_dir
        domain = self.config_obj.domain

        data_f = ind_dir + 'data/' + domain + '/'
        fold_f = ind_dir + 'data/' + domain + '/folds/'
        if not os.path.exists(fold_f):
            os.makedirs(fold_f)
        return data_f, fold_f

    def read_file(self, filename):
        """Reads the appropriate comments file of the domain.
        filename: csv comments file.
        Returns comments dataframe up to the end marker in the config."""
        coms_df = pd.read_csv(filename, lineterminator='\n',
                              nrows=self.config_obj.end)
        return coms_df

    def load_convenience_files(self, coms_df, *args):
        """Reads in helper files, such as dataframes with text similarities
        for each comment, and merges them onto the main dataframe.
        coms_df: dataframe comment subset.
        *args: list of tsv files to read and merge on columns 'com_id'.
        Returns coms_df with merged columns."""
        for arg in args:
            if os.path.exists(arg):
                df = pd.read_csv(arg, lineterminator='\n', sep='\t')
                coms_df = coms_df.merge(df, on='com_id')
        return coms_df

    def split_coms(self, coms_df):
        """Splits the comments into training, validation, and test sets.
        Validation and test sets are the same size.
        coms_df: comments dataframe.
        Returns training and test dataframes based on the training size."""
        coms_df = coms_df[self.config_obj.start:]
        split_ndx1 = int(len(coms_df) * self.config_obj.train_size)
        train_df = coms_df[:split_ndx1]
        split_ndx2 = int(split_ndx1 + ((len(coms_df) - len(train_df)) / 2))
        val_df = coms_df[split_ndx1:split_ndx2]
        test_df = coms_df[split_ndx2:]
        return train_df, val_df, test_df

    def write_folds(self, val_df, test_df, fold_f):
        """Writes validation and test set dataframes to csv files.
        val_df: dataframe with validation set comments.
        test_df: dataframe with test set comments.
        fold_f: folder to save the data to."""
        val_df.to_csv(fold_f + 'val_' + self.config_obj.fold + '.csv',
                      line_terminator='\n', index=None)
        test_df.to_csv(fold_f + 'test_' + self.config_obj.fold + '.csv',
                       line_terminator='\n', index=None)

    def print_subsets(self, train_df, val_df, test_df):
        """Writes basic statistics about the training and test sets.
        train_df: training set comments.
        test_df: test set comments."""
        spam, total = len(train_df[train_df['label'] == 1]), len(train_df)
        percentage = round(self.util_obj.div0(spam, total) * 100, 1)
        s = 'Training set size: ' + str(len(train_df)) + ', '
        s += 'spam: ' + str(spam) + ' (' + str(percentage) + '%)'
        print(s)

        spam, total = len(val_df[val_df['label'] == 1]), len(val_df)
        percentage = round(self.util_obj.div0(spam, total) * 100, 1)
        s = 'Validation set size: ' + str(len(val_df)) + ', '
        s += 'spam: ' + str(spam) + ' (' + str(percentage) + '%)'
        print(s)

        spam, total = len(test_df[test_df['label'] == 1]), len(test_df)
        percentage = round(self.util_obj.div0(spam, total) * 100, 1)
        s = 'Test set size: ' + str(len(test_df)) + ', '
        s += 'spam: ' + str(spam) + ' (' + str(percentage) + '%)'
        print(s)

    # public
    def main(self):
        """Main method that reads in the comments, splits them into train and
        test, writes them to files, and prints out stats.
        Returns the train and test comment dataframes."""
        modified = self.config_obj.modified

        data_f, fold_f = self.define_file_folders()
        coms_filename = self.util_obj.get_comments_filename(modified)
        coms_df = self.read_file(data_f + coms_filename)
        if 'text_id' not in list(coms_df):
            intext_file = data_f + 'intext.tsv'
            coms_df = self.load_convenience_files(coms_df, intext_file)
        train_df, val_df, test_df = self.split_coms(coms_df)
        self.write_folds(val_df, test_df, fold_f)
        self.print_subsets(train_df, val_df, test_df)
        self.classification_obj.main(train_df, val_df, test_df)
        return val_df, test_df
