import pandas as pd


class Data:

    data_dir = 'independent/data/'

    # public
    def __init__(self, gen_obj):
        self.gen_obj = gen_obj

    def get_data(self, start=0, end=1000, domain='twitter'):
        filename = Data.data_dir + domain + '/comments.csv'
        coms_df = pd.read_csv(filename, lineterminator='\n', nrows=end)
        coms_df = coms_df[start:].reset_index().drop(['index'], axis=1)
        return coms_df

    def sep_relational_data(self, df, relations, domain='twitter'):
        domain_dir = Data.data_dir + domain + '/'
        ids = set()
        df = df.copy()

        for relation, group, group_id in relations:
            r_df = self.gen_obj.gen_rel_df(df, group_id, domain_dir)
            ids.update(set(r_df['com_id']))
        df = df[df['com_id'].isin(ids)]
        return df

    def split_data(self, df, train_size=0.7, val_size=0.15):
        num_coms = len(df)
        split_ndx1 = int(num_coms * train_size)
        split_ndx2 = split_ndx1 + int(num_coms * val_size)

        train_df = df[:split_ndx1]
        val_df = df[split_ndx1:split_ndx2]
        test_df = df[split_ndx2:]

        data = {'train': train_df, 'val': val_df, 'test': test_df}
        return data
