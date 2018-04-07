import pandas as pd
import util as ut
from textblob import TextBlob


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
    ut.out('reading in data...')
    df = pd.read_csv(data_dir + '2016_election.csv', lineterminator='\n')
    df = df.drop_duplicates('com_id')
    df.to_csv('2016_election.csv', index=None)


def sentiment(data_dir):
    polarity = lambda x: TextBlob(x).sentiment.polarity
    subjectivity = lambda x: TextBlob(x).sentiment.subjectivity

    df = pd.read_csv(data_dir + 'comments.csv', lineterminator='\n')
    df['polarity'] = df['text'].apply(polarity)
    df['subjectivity'] = df['text'].apply(subjectivity)
    df.to_csv(data_dir + 'comments_new.csv', index=None)


if __name__ == '__main__':
    data_dir = 'independent/data/twitter/'
    sentiment(data_dir)
