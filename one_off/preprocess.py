import os
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


def russia(data_dir, prefix='f1_', out_dir=''):
    pd.set_option('display.width', 181)

    files = os.listdir(data_dir)
    # u_files = [u for u in files if prefix in u and '-users' in u]
    m_files = [m for m in files if prefix in m and '-msgs' in m]

    # u_filter_cols = ['id', 'created_at']
    # u_del_cols = ['default_profile', 'default_profile_image', 'following',
    #               'name', 'screen_name', 'geo_enabled']
    # u_str_cols = ['description', 'location', 'time_zone']
    # u_int_cols = ['id', 'favourites_count', 'followers_count',
    #               'friends_count', 'statuses_count']
    # u_bool_cols = ['verified']

    # u_dfs = []
    # for u_file in u_files:
    #     print(u_file)
    #     u = pd.read_csv(data_dir + u_file, lineterminator='\n')

    #     for col in u_filter_cols:
    #         u = u[~pd.isnull(u[col])]
    #     u = u.drop(u_del_cols, axis=1)
    #     for col in u_str_cols:
    #         u[col] = u[col].fillna('')
    #     for col in u_int_cols:
    #         u[col] = u[col].fillna(0).apply(int)
    #     for col in u_bool_cols:
    #         u[col] = u[col].apply(lambda x: 0 if x is False else 1)
    #     u.columns = ['user_' + x for x in list(u)]
    #     u_dfs.append(u)
    # u_df = pd.concat(u_dfs)
    # u_df.to_csv(out_dir + prefix + 'users.csv', index=None)
    # u_df = None
    # u_dfs = None

    m_filter_cols = ['id', 'created_at', 'user_id', 'lang']
    m_del_cols = ['geo', 'lang', 'place']
    m_str_cols = ['full_text']
    m_int_cols = ['id', 'user_id']

    m_dfs = []
    for m_file in m_files:
        print(m_file)
        m = pd.read_csv(data_dir + m_file, lineterminator='\n')
        if 'user_id' in m:
            m['created_at'] = pd.to_datetime(m['created_at'])
            for col in m_filter_cols:
                m = m[~pd.isnull(m[col])]
            m = m.drop(m_del_cols, axis=1)
            for col in m_str_cols:
                m[col] = m[col].fillna('')
            for col in m_int_cols:
                m[col] = m[col].fillna(0).apply(int)
            del m['created_at']
            m = m.rename(columns={'full_text': 'text', 'id': 'com_id',
                                  'created_df': 'timestamp'})
            m_dfs.append(m)
    m_df = pd.concat(m_dfs)
    m_df.to_csv(out_dir + prefix + 'msgs.csv', index=None)


if __name__ == '__main__':
    data_dir = '/Volumes/Brophy/election/csv/'
    out_dir = '/Volumes/Brophy/election/merged/'

    # adclicks(data_dir=data_dir)
    for prefix in ['f1_', 'f2_', 'f3_', 'f4_', 'f5_']:
        russia(data_dir, prefix=prefix, out_dir=out_dir)
