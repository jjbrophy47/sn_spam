import pandas as pd


class Data:

    data_dir = 'independent/data/'

    # public
    def __init__(self, generator_obj):
        self.gen_obj = generator_obj

    def get_rel_ids(self, df, domain='twitter', relations=[]):
        dd = Data.data_dir + domain + '/'
        df = self.gen_obj.gen_relational_ids(df, relations, data_dir=dd)
        return df

    def get_data(self, start=0, end=1000, domain='twitter'):
        filename = Data.data_dir + domain + '/comments.csv'
        coms_df = pd.read_csv(filename, lineterminator='\n', nrows=end)
        coms_df = coms_df[start:].reset_index().drop(['index'], axis=1)
        return coms_df

    def sep_data(self, df, relations=[], domain='twitter', data='both'):
        if data == 'both':
            return df

        ids = set()
        for relation, group, group_id in relations:
            q_df = df[df[group_id] != []]
            ids.update(set(q_df['com_id']))

        ind_df = df[~df['com_id'].isin(ids)]
        rel_df = df[df['com_id'].isin(ids)]

        result_df = ind_df if data == 'ind' else rel_df
        return result_df

    def split_data(self, df, train_size=0.7, val_size=0.15):
        num_coms = len(df)
        split_ndx1 = int(num_coms * train_size)
        split_ndx2 = split_ndx1 + int(num_coms * val_size)

        train_df = df[:split_ndx1]
        val_df = df[split_ndx1:split_ndx2]
        test_df = df[split_ndx2:]

        data = {'train': train_df, 'val': val_df, 'test': test_df}
        lens = (len(train_df), len(val_df), len(test_df))
        print('train: %d, val: %d, test: %d' % lens)
        return data
