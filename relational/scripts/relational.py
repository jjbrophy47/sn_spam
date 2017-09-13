"""
This module generates predicates for and runs the relational model.
"""
import os
import pandas as pd


class Relational:
    """Class that handles all operations pertaining to the relational
    model."""

    def __init__(self, config_obj, psl_obj, tuffy_obj):
        """Initialize all object dependencies for this class."""

        self.config_obj = config_obj
        """User settings."""
        self.psl_obj = psl_obj
        """Object to run the relational model using PSL."""
        self.tuffy_obj = tuffy_obj
        """Object to run the relational model using Tuffy."""

    # public
    def compile_reasoning_engine(self):
        """Uses the psl module to compile the PSL and groovy scripts."""
        psl_f, _, _, _, _ = self.define_file_folders()
        self.psl_obj.compile(psl_f)

    def main(self, val_df, test_df):
        """Sets up file structure, merges predictions, and runs the model.
        val_df: validation dataframe.
        test_df: testing dataframe."""
        psl_f, tuffy_f, data_f, folds_f, pred_f = self.define_file_folders()
        val_df, test_df = self.check_dataframes(val_df, test_df, folds_f)
        val_df = self.merge_ind_preds(val_df, 'val', pred_f)
        test_df = self.merge_ind_preds(test_df, 'test', pred_f)
        self.run_relational_model(val_df, test_df, data_f, psl_f, tuffy_f)

    # private
    def define_file_folders(self):
        """Returns absolute path directories."""
        ind_dir = self.config_obj.ind_dir
        rel_dir = self.config_obj.rel_dir
        domain = self.config_obj.domain

        psl_f = rel_dir + 'psl/'
        tuffy_f = rel_dir + 'tuffy/'
        rel_data_f = rel_dir + 'data/' + domain + '/'
        folds_f = ind_dir + 'data/' + domain + '/folds/'
        pred_f = ind_dir + 'output/' + domain + '/predictions/'
        if not os.path.exists(rel_data_f):
            os.makedirs(rel_data_f)
        return psl_f, tuffy_f, rel_data_f, folds_f, pred_f

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

    def run_psl(self, val_df, test_df, rel_data_f, psl_f):
        """Sets up everything needed to run the psl relational model.
        val_df: validation dataframe.
        test_df: testing dataframe.
        rel_data_f: relational data foler.
        psl_f: psl folder."""
        self.psl_obj.clear_data(rel_data_f)

        print('Relational predicates:')
        self.psl_obj.gen_predicates(val_df, 'val', rel_data_f)
        self.psl_obj.gen_predicates(test_df, 'test', rel_data_f)
        self.psl_obj.gen_model(rel_data_f)
        self.psl_obj.run(psl_f)

    def run_tuffy(self, val_df, test_df, rel_data_f, tuffy_f):
        """Sets up everything needed to run the tuffy relational model.
        val_df: validation dataframe.
        test_df: testing dataframe.
        rel_data_f: relational data foler.
        tuffy_f: tuffy folder."""
        self.tuffy_obj.clear_data(tuffy_f)

        print('Relational predicates:')
        self.tuffy_obj.gen_predicates(val_df, 'val', tuffy_f)
        self.tuffy_obj.gen_predicates(test_df, 'test', tuffy_f)
        self.tuffy_obj.run(tuffy_f)
        pred_df = self.tuffy_obj.parse_output(tuffy_f)
        self.tuffy_obj.evaluate(test_df, pred_df)

    def run_relational_model(self, val_df, test_df, rel_data_f, psl_f,
            tuffy_f):
        """Runs the appropriate reasoning engine based on user settings.
        val_df: validation dataframe.
        test_df: testing dataframe.
        rel_data_f: relational data folder.
        psl_f: psl folder.
        tuffy_f: tuffy folder."""
        if self.config_obj.engine == 'psl':
            self.run_psl(val_df, test_df, rel_data_f, psl_f)
        elif self.config_obj.engine == 'tuffy':
            self.run_tuffy(val_df, test_df, rel_data_f, tuffy_f)
