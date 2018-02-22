import re
import httplib2
import pandas as pd


def lengthen_urls(df, col='text', regex_str=r'(http[^\s]+)', out_dir='',
        fname='comments.csv'):

    h = httplib2.Http('.cache')
    regex = re.compile(regex_str)

    for index, string in list(zip(list(df.index), list(df[col]))):
        short_urls = regex.findall(string)

        for short_url in short_urls:
            try:
                long_url = h.request(short_url)[0]['content-location']
                df.at[index, col] = df.at[index, col].replace(short_url,
                        long_url)
            except (httplib2.ServerNotFoundError, httplib2.RelativeURIError):
                pass

    df.to_csv(out_dir + fname, index=None)


if __name__ == '__main__':
    df = pd.read_csv('independent/data/twitter/comments.csv')
    lengthen_urls(df, fname='replace.csv')
