"""
Module containing the Independent class to handle all operations pertaining
to the independent model.
"""
import os
import pandas as pd


class Independent:

    def __init__(self, config_obj, classification_obj, generator_obj,
                 util_obj):
        self.config_obj = config_obj
        self.classification_obj = classification_obj
        self.gen_obj = generator_obj
        self.util_obj = util_obj

    # public
    def main(self, data):
        """Main method that reads in the comments, splits them into train and
        test, writes them to files, and prints out stats.
        Returns the train and test comment dataframes."""
        self.util_obj.start()
        data_f, fold_f, status_f = self.file_folders()
        sw = self.open_status_writer(status_f)

        # relations = self.config_obj.relations
        train_df, val_df, test_df = data['train'], data['val'], data['test']

        # TODO: update this method to work with lists of rel_ids.
        if self.config_obj.alter_user_ids:
            val_df = self.alter_user_ids(coms_df, val_df)
            test_df = self.alter_user_ids(coms_df, test_df)

        if self.config_obj.separate_relations:
            val_df = self.separate_relations(train_df, val_df)
            test_df = self.separate_relations(train_df, test_df)

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

        val_df = val_df.reset_index().drop(['index'], axis=1)
        test_df = test_df.reset_index().drop(['index'], axis=1)
        return val_df, test_df

    # private
    def file_folders(self):
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
        fold = self.config_obj.fold
        fname = status_f + 'ind_' + fold + '.txt'
        f = self.util_obj.open_writer(fname)
        return f

    # TODO: update to accommodate user_ids as a list
    def alter_user_ids(self, coms_df, test_df):
        max_user_id = coms_df['user_id'].max() + 1
        user_ids = list(zip(test_df['label'], test_df['user_id']))
        new_user_ids = []

        for label, user_id in user_ids:
            new_user_ids.append(max_user_id if label == 1 else user_id)
            max_user_id += 1
        test_df['user_id'] = new_user_ids
        return test_df

    def _separate(self, te_ids, tr_id, new_id):
        if tr_id in te_ids:
            te_ids.remove(tr_id)
            te_ids.append(new_id)
        return te_ids

    def separate_relations(self, coms_df, train_df, test_df):
        for relation, group, g_id in self.config_obj.relations:
            tr_ids = {x for sublist in list(train_df[g_id]) for x in sublist}
            te_ids = {x for sublist in list(test_df[g_id]) for x in sublist}

            max_id = max(max(tr_ids), max(te_ids))

            for tr_id in tr_ids.intersection(te_ids):
                max_id += 1
                args = (tr_id, max_id)
                test_df[g_id] = test_df[g_id].apply(self._separate, args=args)

            return test_df

    def write_folds(self, val_df, test_df, fold_f):
        fold = self.config_obj.fold
        val_fname = fold_f + 'val_' + fold + '.csv'
        test_fname = fold_f + 'test_' + fold + '.csv'

        val_df.to_csv(val_fname, line_terminator='\n', index=None)
        test_df.to_csv(test_fname, line_terminator='\n', index=None)

    def print_subsets(self, train_df, val_df, test_df, fw=None):
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
