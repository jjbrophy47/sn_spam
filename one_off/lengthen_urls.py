import re
import sys
import httplib2
import pandas as pd


# public
def lengthen_urls(df, c='text', regex_str=r'(http[^\s]+)', out_dir='',
                  fname='comments.csv'):

    h = httplib2.Http('.cache', timeout=1)
    regex = re.compile(regex_str)

    msgs = list(zip(list(df.index), list(df[c])))

    for i, (n, string) in enumerate(msgs):
        # if i % 1000 == 0:
        _out('(%d/%d)' % (i, len(msgs)))

        short_urls = regex.findall(string)

        for short_url in short_urls:
            try:
                header = h.request(short_url)[0]
                if 'content-location' in header:
                    long_url = header['content-location']
                    df.at[n, c] = df.at[n, c].replace(short_url, long_url)
                    _out('%s -> %s' % (short_url, long_url))
                else:
                    _out('ERR: %s' % short_url)
            except Exception:
                _out('ERR: %s' % short_url)

    df = df[['com_id', 'text']]
    df.to_csv(out_dir + fname, index=None)


# private
def _out(message=''):
    sys.stdout.write(message + '\n')
    sys.stdout.flush()


if __name__ == '__main__':
    in_dir = 'independent/data/twitter/'

    _out('reading in messages...')
    df = pd.read_csv(in_dir + 'comments.csv')
    df = df[df['text'].str.contains('http')]
    lengthen_urls(df, out_dir=in_dir, fname='link.csv')
