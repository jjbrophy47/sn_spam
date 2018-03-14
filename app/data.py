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

    def get_data(self, start=0, end=1000, domain='twitter', evaluation='cc'):
        skiprows = range(1, start)
        nrows = end - start

        if evaluation == 'tt':
            train_path = Data.data_dir + domain + '/train.csv'
            test_path = Data.data_dir + domain + '/test.csv'
            train_df = pd.read_csv(train_path, lineterminator='\n',
                                   skiprows=skiprows, nrows=nrows)
            train_df = train_df.reset_index().drop(['index'], axis=1)
            test_df = pd.read_csv(test_path, lineterminator='\n', nrows=end)
            return train_df, test_df

        elif evaluation == 'cc':
            path = Data.data_dir + domain + '/comments.csv'
            coms_df = pd.read_csv(path, lineterminator='\n',
                                  skiprows=skiprows, nrows=nrows)
            coms_df = coms_df.reset_index().drop(['index'], axis=1)
            return coms_df

    def sep_data(self, df, relations=[], domain='twitter', data='both'):
        if data == 'both':
            return df

        ids = set()
        list_filter = lambda x: True if x != [] else False
        for relation, group, group_id in relations:
            q_df = df[df[group_id].apply(list_filter)]
            ids.update(set(q_df['com_id']))

        ind_df = df[~df['com_id'].isin(ids)]
        rel_df = df[df['com_id'].isin(ids)]

        result_df = ind_df if data == 'ind' else rel_df
        return result_df

    def split_data(self, df, train_size=0.7, val_size=0.15):
        num_coms = len(df)

        if train_size == 0 and val_size is None:
            data = {'train': df, 'val': None, 'test': None}
            print('train: %d' % len(df))

        elif val_size is None:
            split_ndx = int(num_coms * train_size)
            train_df = df[:split_ndx]
            val_df = df[split_ndx:]
            data = {'train': train_df, 'val': val_df, 'test': None}
            lens = (len(train_df), len(val_df))
            print('train: %d, val: %d' % lens)

        else:
            split_ndx1 = int(num_coms * train_size)
            split_ndx2 = split_ndx1 + int(num_coms * val_size)

            train_df = df[:split_ndx1]
            val_df = df[split_ndx1:split_ndx2]
            test_df = df[split_ndx2:]

            data = {'train': train_df, 'val': val_df, 'test': test_df}
            lens = (len(train_df), len(val_df), len(test_df))
            print('train: %d, val: %d, test: %d' % lens)
        return data
