import re
import sys
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def similarities(df, num_chunks=10, target_col='text', out_col='text_id',
                 out_dir='', fname='sim.csv'):
    out('extracting messages...')

    if num_chunks == 1:
        max_id = 0
        df = df.reset_index().drop(['index'], axis=1)
        strings = list(df[target_col])
        sim_df, max_id = find_similarities(df, strings, max_id=max_id,
                                           output_col=out_col)

    else:
        df['len'] = df[target_col].str.len()
        df['len_id'] = pd.qcut(df['len'], num_chunks,
                               duplicates='drop').cat.codes

        sim_chunks = []
        max_id = 0
        for len_id in range(df['len_id'].min(), df['len_id'].max()):
            out('\nlen_id: %d' % (len_id))
            df_chunk = df[df['len_id'] == len_id]
            df_chunk = df_chunk.reset_index().drop(['index'], axis=1)
            strings = list(df_chunk[target_col])
            sim_chunk_df, max_text_id = find_similarities(df_chunk, strings,
                                                          max_id=max_id,
                                                          output_col=out_col)
            sim_chunks.append(sim_chunk_df)
        sim_df = pd.concat(sim_chunks)

    sim_df.to_csv(out_dir + fname, index=None)
    out(str(sim_df))


def out(message=''):
    sys.stdout.write(message + '\n')
    sys.stdout.flush()


def ngrams(string, n=3):
    string = re.sub(r'[,-./]|\sBD', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def tf_idf(strings, analyzer='word'):
    vectorizer = TfidfVectorizer(min_df=1, analyzer=analyzer)
    tf_idf_matrix = vectorizer.fit_transform(strings)
    return tf_idf_matrix


def find_similarities(df, strings, sim_thresh=0.8, max_id=0,
                      output_col='text_id'):
    out('creating tf-idf matrix...')
    tf_idf_matrix = tf_idf(strings, analyzer=ngrams)

    out('computing cosine similarities...')
    cos_sim = cosine_similarity(tf_idf_matrix)

    out('filtering out simiarities below threshold...')
    cos_sim[cos_sim < sim_thresh] = 0.0

    out('converting matrix to sparse matrix...')
    scm = sparse.csr_matrix(cos_sim)

    out('putting matches into groups...')
    groups = {-1: set()}
    i = max_id + 1

    for ndx in range(len(strings)):
        matches = set(scm[ndx].indices)
        group = set(matches)

        if len(group) > 1:
            groups[i] = group
            i += 1
        else:
            groups[-1].update(group)

    rows = []
    indices = []
    for group_id, ndxs in groups.items():
        indices.extend(ndxs)
        rows.extend([group_id] * len(ndxs))

    r = []
    msgs = list(zip(indices, rows))
    for ndx, group_id in msgs:
        msg_id = df.loc[ndx]['com_id']
        r.append((msg_id, group_id))

    temp_df = pd.DataFrame(r, columns=['com_id', output_col], index=indices)
    temp_df = temp_df.sort_index()
    temp_df = temp_df[temp_df[output_col] != -1]
    max_id = max(temp_df[output_col])
    return temp_df, max_id

if __name__ == '__main__':
    # df = pd.read_csv('independent/data/toxic/comments.csv', nrows=None)
    df = pd.read_csv('independent/data/youtube/comments.csv', nrows=None)
    similarities(df, num_chunks=150, target_col='text', output_col='text_id')
