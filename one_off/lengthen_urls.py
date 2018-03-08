import re
import sys
import httplib2
import argparse
import pandas as pd


def out(message=''):
    sys.stdout.write(message + '\n')
    sys.stdout.flush()


def lengthen_urls(df, c='text', regex_str=r'(http[^\s]+)', out_dir='',
                  fname='comments.csv'):

    h = httplib2.Http('.cache')
    regex = re.compile(regex_str)

    msgs = list(zip(list(df.index), list(df[c])))

    for i, (n, string) in enumerate(msgs):
        if i % 1000 == 0:
            out('(%d/%d)' % (i, len(msgs)))

        short_urls = regex.findall(string)

        for short_url in short_urls:
            try:
                header = h.request(short_url)[0]
                if 'content-location' in header:
                    long_url = header['content-location']
                    df.at[n, c] = df.at[n, c].replace(short_url, long_url)
                    print(short_url, long_url)
                else:
                    print('\t' + short_url)
            except Exception:
                print('\t' + short_url)

    df.to_csv(out_dir + fname, index=None)


if __name__ == '__main__':
    description = 'Tool to replace shortened urls with their original urls.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', help='csv number to process', type=int)
    args = parser.parse_args()

    df = pd.read_csv('chunk_' + str(args.i) + '.csv')
    lengthen_urls(df, fname='replace_' + str(args.i) + '.csv')
