import re
import sys
import pandas as pd


def extract(df, target_col='text', info_type='link', out_dir=''):
    d, i = {}, 0
    regex = _get_regex(info_type)

    df = df[['com_id', target_col]]
    for ndx, com_id, text in df.itertuples():
        i += 1
        if i % 100000 == 0:
            _out('(%d/%d)...' % (i, len(df)))

        info = _get_items(text, regex)
        if info != '':
            d[com_id] = info

    if len(d) > 0:
        info_df = pd.DataFrame.from_dict(d, orient='index').reset_index()
        info_df.columns = ['com_id', info_type + '_id']
        fname = info_type + '.csv'
        _out(str(info_df))
        _out('writing info to csv...')
        info_df.to_csv(out_dir + fname, index=None)
    else:
        _out('No extractions made...')


# private
def _out(message=''):
    sys.stdout.write(message + '\n')
    sys.stdout.flush()


def _get_regex(info_type='link'):
    d = {}
    d['hashtag'] = re.compile(r'(#\w+)')
    d['mention'] = re.compile(r'(@\w+)')
    d['link'] = re.compile(r'(http[^\s]+)')
    return d[info_type]


def _get_items(text, regex, str_form=True):
    items = regex.findall(text)
    result = sorted([x.lower() for x in items])
    if str_form:
        result = ''.join(result)
    return result

if __name__ == '__main__':
    domain = 'youtube'
    data_dir = 'independent/data/' + domain + '/'

    _out('reading in data...')
    df = pd.read_csv(data_dir + 'comments.csv')
    extract(df, target_col='text', info_type='hashtag', out_dir=data_dir)
