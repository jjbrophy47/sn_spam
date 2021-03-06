"""
This module generates predicates for and runs the relational model.
"""
import os
import pandas as pd


class Relational:
    """Class that handles all operations pertaining to the relational
    model."""

    def __init__(self, config_obj, psl_obj, tuffy_obj, util_obj):
        """Initialize all object dependencies for this class."""

        self.config_obj = config_obj
        """User settings."""
        self.psl_obj = psl_obj
        """Object to run the relational model using PSL."""
        self.tuffy_obj = tuffy_obj
        """Object to run the relational model using Tuffy."""
        self.util_obj = util_obj
        """General utility methods."""

    # public
    def compile_reasoning_engine(self):
        """Uses the psl module to compile the PSL and groovy scripts."""
        psl_f, _, _, _, _, _ = self.file_folders()
        self.psl_obj.compile(psl_f)

    def main(self, val_df, test_df):
        """Sets up file structure, merges predictions, and runs the model.
        val_df: validation dataframe.
        test_df: testing dataframe."""
        psl_f, psl_d_f, tuffy_f, folds_f, pred_f, sts_f = self.file_folders()
        sw = self.open_status_writer(sts_f)

        self.util_obj.start(fw=sw)
        val_df, test_df = self.check_dataframes(val_df, test_df, folds_f)
        val_df = self.merge_ind_preds(val_df, 'val', pred_f)
        test_df = self.merge_ind_preds(test_df, 'test', pred_f)

        self.run_relational_model(val_df, test_df, psl_f, psl_d_f, tuffy_f,
                fw=sw)

        sw = self.open_status_writer(sts_f, mode='a')
        self.util_obj.end('total relational model time: ', fw=sw)
        self.util_obj.close_writer(sw)

    # private
    def file_folders(self):
        """Returns absolute path directories."""
        ind_dir = self.config_obj.ind_dir
        rel_dir = self.config_obj.rel_dir
        domain = self.config_obj.domain

        psl_f = rel_dir + 'psl/'
        psl_data_f = psl_f + 'data/' + domain + '/'
        tuffy_f = rel_dir + 'tuffy/'
        folds_f = ind_dir + 'data/' + domain + '/folds/'
        pred_f = ind_dir + 'output/' + domain + '/predictions/'
        status_f = rel_dir + 'output/' + domain + '/status/'
        if not os.path.exists(psl_data_f):
            os.makedirs(psl_data_f)
        if not os.path.exists(status_f):
            os.makedirs(status_f)
        return psl_f, psl_data_f, tuffy_f, folds_f, pred_f, status_f

    def open_status_writer(self, status_f, mode='w'):
        """Open a file writer to write status updates to.
        status_f: status folder.
        mode: write or append to file.
        Return file writer."""
        fold = self.config_obj.fold
        operation = 'infer' if self.config_obj.infer else 'train'
        fname = status_f + operation + '_' + fold + '.txt'
        f = self.util_obj.open_writer(fname, mode)
        return f

    def check_dataframes(self, val_df, test_df, folds_f):
        """If dataframes are None, then read them in.
        val_df: validation dataframe.
        test_df: testing dataframe.
        folds_f: folds folder.
        Returns dataframes for both datasets."""
        fold = self.config_obj.fold

        if val_df is None or test_df is None:
            val_df = pd.read_csv(folds_f + 'val_' + fold + '.csv')
            test_df = pd.read_csv(folds_f + 'test_' + fold + '.csv')
        return val_df, test_df

    def merge_ind_preds(self, df, dset, ind_pred_f):
        """Merge the independent model predictions with the relational ones.
        df: dataframe containing comments.
        dset: dataset (e.g. 'val', 'test')
        ind_pred_f: folder where the independent predictions are stored.
        Returns merged dataframe."""
        fold = self.config_obj.fold
        preds_df = pd.read_csv(ind_pred_f + dset + '_' + fold + '_preds.csv')
        df = df.merge(preds_df)
        return df

    def run_psl(self, val_df, test_df, psl_f, psl_data_f, fw=None):
        """Sets up everything needed to run the psl relational model.
        val_df: validation dataframe.
        test_df: testing dataframe.
        psl_f: psl folder.
        psl_data_f: relational data foler."""
        self.psl_obj.clear_data(psl_data_f, fw=fw)

        self.util_obj.start('\nbuilding predicates...', fw=fw)
        self.psl_obj.gen_predicates(val_df, 'val', psl_data_f, fw=fw)
        self.psl_obj.gen_predicates(test_df, 'test', psl_data_f, fw=fw)
        self.psl_obj.gen_model(psl_data_f)
        self.psl_obj.network_size(psl_data_f, fw=fw)
        self.util_obj.end('\n\ttime: ', fw=fw)
        self.util_obj.close_writer(fw)
        self.psl_obj.run(psl_f)

    def run_tuffy(self, val_df, test_df, tuffy_f, fw=None):
        """Sets up everything needed to run the tuffy relational model.
        val_df: validation dataframe.
        test_df: testing dataframe.
        tuffy_f: tuffy folder."""
        self.tuffy_obj.clear_data(tuffy_f)

        self.util_obj.start('\nbuilding predicates...', fw=fw)
        self.tuffy_obj.gen_predicates(val_df, 'val', tuffy_f)
        self.tuffy_obj.gen_predicates(test_df, 'test', tuffy_f)
        self.util_obj.end('\n\ttime: ', fw=fw)
        self.tuffy_obj.run(tuffy_f)
        pred_df = self.tuffy_obj.parse_output(tuffy_f)
        self.tuffy_obj.evaluate(test_df, pred_df)

    def run_relational_model(self, val_df, test_df, psl_f, psl_data_f,
            tuffy_f, fw=None):
        """Runs the appropriate reasoning engine based on user settings.
        val_df: validation dataframe.
        test_df: testing dataframe.
        psl_f: psl folder.
        psl_data_f: relational data folder.
        tuffy_f: tuffy folder."""
        if self.config_obj.engine == 'psl':
            self.run_psl(val_df, test_df, psl_f, psl_data_f, fw=fw)
        elif self.config_obj.engine == 'tuffy':
            self.run_tuffy(val_df, test_df, tuffy_f, fw=fw)
