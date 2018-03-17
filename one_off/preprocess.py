import pandas as pd


def adclicks(data_dir=''):
    data_dir += 'adclicks/'

    df = pd.read_csv(data_dir + 'test.csv')
    df = df.rename(columns={'click_id': 'com_id'})
    df.to_csv(data_dir + 'test.csv', index=None)

    max_id = df['com_id'].max() + 1

    df = pd.read_csv(data_dir + 'train.csv')
    df = df.rename(columns={'is_attributed': 'label'})
    df['com_id'] = range(max_id, max_id + len(df))
    df.to_csv(data_dir + 'train.csv', index=None)


def russia(data_dir):
    print('reading russia data...')
    ru = pd.read_csv(data_dir + 'russia_users.csv').drop_duplicates()
    rm = pd.read_csv(data_dir + 'russia_msgs.csv')
    rf = rm.merge(ru)
    rf['label'] = 1

    efs = []
    prefixes = ['f1_', 'f2_', 'f3_', 'f4_', 'f5_', 'f6_', 'f7_', 'f8_', 'f9_']
    for prefix in prefixes:
        print('election chunk: %s data...' % prefix)
        print('reading user data...')
        eu = pd.read_csv(data_dir + prefix + 'users.csv').drop_duplicates()
        print('reading msg data...')
        em = pd.read_csv(data_dir + prefix + 'msgs.csv')
        print('merging data...')
        ef = em.merge(eu)
        ef['label'] = 0
        efs.append(ef)

    print('concatenating election data...')
    ef = pd.concat(efs)
    print('filtering...')
    ef = ef[~ef['com_id'].isin(rf['com_id'])]
    print('concatenating with russian data...')
    df = pd.concat([ef, rf])
    print('sorting by com_id...')
    df = df.sort_values('com_id')
    print('writing to csv...')
    df.to_csv(data_dir + '2016_election.csv', index=None)

if __name__ == '__main__':
    out_dir = 'independent/data/russia/csv/'
    russia(out_dir)
