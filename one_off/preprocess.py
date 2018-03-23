import pandas as pd
import util as ut


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


def russia_msgs(data_dir):
    ut.out('Msgs...')
    prefixes = ['f1_', 'f2_', 'f3_', 'f4_', 'f5_', 'f6_', 'f7_', 'f8_', 'f9_']

    for prefix in prefixes:
        ut.out('prefix: %s' % prefix)
        m = pd.read_csv(data_dir + prefix + 'msgs.csv', lineterminator='\n')
        ut.out(str(len(m)))
        m = m[~pd.isnull(m['user_id'])]
        m = m[~pd.isnull(m['com_id'])]
        m = m[~pd.isnull(m['text'])]
        m['user_id'] = m['user_id'].astype(int)
        m['com_id'] = m['com_id'].astype(int)
        m = m.drop_duplicates()
        ut.out(str(len(m)))
        m.to_csv(data_dir + prefix + 'msgs_g.csv', index=None)


def russia_users(data_dir):
    ut.out('Users...')
    prefixes = ['f1_', 'f2_', 'f3_', 'f4_', 'f5_', 'f6_', 'f7_', 'f8_', 'f9_']

    for prefix in prefixes:
        ut.out('prefix: %s' % prefix)
        u = pd.read_csv(data_dir + prefix + 'users.csv', lineterminator='\n')
        ut.out(str(len(u)))
        u = u[~pd.isnull(u['user_created_at'])]
        u = u[~pd.isnull(u['user_id'])]
        u['user_description'] = u['user_description'].fillna(' ')
        u['user_favourites_count'] = u['user_favourites_count'].fillna(0)
        u['user_followers_count'] = u['user_followers_count'].fillna(0)
        u['user_friends_count'] = u['user_friends_count'].fillna(0)
        u['user_location'] = u['user_location'].fillna(' ')
        u['user_statuses_count'] = u['user_statuses_count'].fillna(0)
        u['user_time_zone'] = u['user_time_zone'].fillna(' ')
        u['user_verified'] = u['user_verified'].fillna(0)
        u['user_favourites_count'] = u['user_favourites_count'].apply(int)
        u['user_followers_count'] = u['user_followers_count'].apply(int)
        u['user_friends_count'] = u['user_friends_count'].apply(int)
        u['user_statuses_count'] = u['user_statuses_count'].apply(int)
        u['user_verified'] = u['user_verified'].apply(int)
        u = u.drop_duplicates()
        ut.out(str(len(u)))
        u.to_csv(data_dir + prefix + 'users_g.csv', index=None)


def russia(data_dir):
    print('reading russia data...')
    ru = pd.read_csv(data_dir + 'russia_users.csv', lineterminator='\n')
    ru = ru.drop_duplicates()
    rm = pd.read_csv(data_dir + 'russia_msgs.csv', lineterminator='\n')
    rm = rm.drop_duplicates()
    rf = rm.merge(ru)
    rf['label'] = 1

    prefixes = ['f1_', 'f2_', 'f3_', 'f4_', 'f5_', 'f6_', 'f7_', 'f8_', 'f9_']

    mfs = []
    ut.out('reading msg data...')
    for prefix in prefixes:
        ut.out('%s...' % prefix)
        mf = pd.read_csv(data_dir + prefix + 'msgs_g.csv',
                         lineterminator='\n')
        mf['label'] = 0
        mfs.append(mf)
    mf = pd.concat(mfs)
    mf = mf.drop_duplicates()

    ufs = []
    ut.out('reading user data...')
    for prefix in prefixes:
        ut.out('%s' % prefix)
        uf = pd.read_csv(data_dir + prefix + 'users_g.csv',
                         lineterminator='\n')
        ufs.append(uf)
    uf = pd.concat(ufs)
    uf = uf.drop_duplicates()

    ut.out('merging msgs and users...')
    ef = mf.merge(uf)

    ut.out('filtering...')
    ef = ef[~ef['com_id'].isin(rf['com_id'])]

    ut.out('concatenating with russian data...')
    df = pd.concat([ef, rf])

    ut.out('sorting by com_id...')
    df = df.sort_values('com_id')

    ut.out('writing to csv...')
    df.to_csv(data_dir + '2016_election.csv', index=None)

if __name__ == '__main__':
    out_dir = 'independent/data/russia/'
    russia(out_dir)
    # russia_msgs(out_dir)
    # russia_users(out_dir)
