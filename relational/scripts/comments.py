"""
Spam comments to be classified by the relational model.
"""
import os
import numpy as np


class Comments:
    """Class to write comment predicate data for the relational model."""

    def __init__(self, config_obj, util_obj):
        """Initializes object dependencies."""

        self.config_obj = config_obj
        """User settings."""
        self.util_obj = util_obj
        """Utility methods."""

    # public
    def build(self, df, dset, data_f=None, tuffy=False):
        """Writes predicate info to the designated data folder.
        df: comments dataframe.
        dset: dataset (e.g. val or test).
        data_f: data folder to save predicate files.
        tuffy: boolean indicating if tuffy is the engine being used."""
        if data_f is None:
            data_f = self.define_file_folders()
        unique_df = self.drop_duplicate_comments(df)

        if tuffy:
            self.write_tuffy_predicates(unique_df, dset, data_f)
        else:
            self.write_predicates(unique_df, dset, data_f)

    # private
    def define_file_folders(self):
        """Returns absolute path directories."""
        rel_dir = self.config_obj.rel_dir
        domain = self.config_obj.domain

        data_f = rel_dir + 'data/' + domain + '/'
        if not os.path.exists(data_f):
            os.makedirs(data_f)
        return data_f

    def drop_duplicate_comments(self, df):
        """Filter out any duplicate comments using the specified columns.
        df: comments dataframe.
        Returns dataframe with unique rows."""
        temp_df = df.filter(['com_id', 'ind_pred', 'label'], axis=1)
        unique_df = temp_df.drop_duplicates()
        return unique_df

    def write_predicates(self, df, dset, data_f):
        """Writes dataframes to csv files.
        df: comments dataframe.
        dset: dataset (e.g. val, test).
        data_f: folder to save data to."""
        fold = self.config_obj.fold
        df['prior'] = 0

        df.to_csv(data_f + dset + '_no_label_' + fold + '.tsv',
                columns=['com_id'], sep='\t', header=None, index=None)
        df.to_csv(data_f + dset + '_' + fold + '.tsv',
                columns=['com_id', 'label'], sep='\t', header=None, index=None)
        df.to_csv(data_f + dset + '_pred_' + fold + '.tsv',
                columns=['com_id', 'ind_pred'], sep='\t', header=None,
                index=None)

    def write_tuffy_predicates(self, df, dset, data_f):
        """Writes predicate data for tuffy.
        df: comments dataframe.
        dset: dataset (e.g. 'val', 'test').
        data_f: folder to store data."""
        ev = open(data_f + dset + '_evidence.txt', 'w')
        q = open(data_f + dset + '_query.txt', 'w')

        for index, row in df.iterrows():
            pred = row.ind_pred
            com_id = str(int(row.com_id))
            wgt = str(np.log(self.util_obj.div0(pred, (1 - pred))))
            ev.write('Indpred(' + com_id + ', ' + wgt + ')\n')
            q.write('Spam(' + com_id + ')\n')

        ev.close()
        q.close()
