import pandas as pd


def split(df, num_chunks=80):

    chunk_size = int(len(df) / 80)
    chunk_start = 0

    for i in range(num_chunks):
        chunk_df = df[chunk_start: chunk_start + chunk_size]
        chunk_df.to_csv('chunk_' + str(i) + '.csv', index=None)
        chunk_start += chunk_size


def combine(in_dir='', in_fname='replace_', num_chunks=80, out_dir='',
        out_fname='comments.csv'):
    chunks = []

    for i in range(num_chunks):
        chunks_df = pd.read_csv(in_dir + in_fname + str(i) + '.csv')
        chunks.append(chunks_df)

    df = pd.concat(chunks)
    df.to_csv(out_dir + out_fname, index=None)


if __name__ == '__main__':
    df = pd.read_csv('independent/data/twitter/comments.csv')
    split(df, num_chunks=80)
    combine(in_dir='chunks/', num_chunks=80)
