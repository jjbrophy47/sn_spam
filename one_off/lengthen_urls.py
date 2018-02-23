import re
import httplib2
import pandas as pd
from socket import timeout


def lengthen_urls(df, c='text', regex_str=r'(http[^\s]+)', out_dir='',
        fname='comments.csv'):

    h = httplib2.Http('.cache')
    regex = re.compile(regex_str)
    errors = (httplib2.ServerNotFoundError, httplib2.RelativeURIError,
            httplib2.RedirectLimit, timeout, ValueError)

    for n, string in list(zip(list(df.index), list(df[c]))):
        short_urls = regex.findall(string)

        for short_url in short_urls:
            try:
                header = h.request(short_url)[0]
                if 'content-location' in header:
                    long_url = header['content-location']
                    df.at[n, c] = df.at[n, c].replace(short_url, long_url)
                    print(long_url)
                else:
                    print('\t' + short_url)
            except errors:
                print('\t' + short_url)

    df.to_csv(out_dir + fname, index=None)


if __name__ == '__main__':
    df = pd.read_csv('independent/data/twitter/comments.csv')
    lengthen_urls(df, fname='replace.csv')
