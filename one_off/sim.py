import re
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# public
def similarities(df, num_chunks=10, target_col='text', out_col='text_id',
                 out_dir='', fname='sim.csv'):
    _out('extracting messages...')

    if num_chunks == 1:
        max_id = 0
        df = df.reset_index().drop(['index'], axis=1)
        strings = list(df[target_col])
        sim_df, max_id = cosine_similarities(df, strings, max_id=max_id,
                                             output_col=out_col)

    else:
        df['len'] = df[target_col].str.len()
        df['len_id'] = pd.qcut(df['len'], num_chunks,
                               duplicates='drop').cat.codes

        sim_chunks = []
        max_id = 0
        for len_id in range(df['len_id'].min(), df['len_id'].max()):
            df_chunk = df[df['len_id'] == len_id]
            _out('\nlen_id: %d, size: %d' % (len_id, len(df_chunk)))

            df_chunk = df_chunk.reset_index().drop(['index'], axis=1)
            strings = list(df_chunk[target_col])
            sim_chunk_df, max_text_id = cosine_similarities(df_chunk, strings,
                                                            max_id=max_id,
                                                            output_col=out_col)
            sim_chunks.append(sim_chunk_df)
        sim_df = pd.concat(sim_chunks)

    sim_df.to_csv(out_dir + fname, index=None)
    _out(str(sim_df))


def knn_similarities(df, sim_thresh=0.8, n_neighbors=100,
                     approx_datapoints=120000, in_col='text',
                     out_col='text_id'):
    _out('splitting data into manageable chunks...')
    dfs = _split_data(df, approx_datapoints=approx_datapoints, in_col=in_col)
    groups = defaultdict(lambda: [])
    group_id = 0

    for n, chunk_df in enumerate(dfs):
        _out('creating tf-idf matrix for chunk %d...' % n)
        g_df = chunk_df.groupby(in_col).size().reset_index()
        strings = list(g_df[in_col])
        tf_idf_matrix = _tf_idf(strings, analyzer=_ngrams)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(tf_idf_matrix)

        _out('querying/filtering each object for its closest neighbors...')
        for row in range(len(strings)):
            distances, indexes = nbrs.kneighbors(tf_idf_matrix.getrow(row))
            nbs = list(zip(distances[0], indexes[0]))
            nbs = [(d, i) for d, i in nbs if d <= sim_thresh]

            _out('\n%s' % strings[row])
            for d, i in nbs[:5]:
                _out('[%d] %s: %f' % (i, strings[i], d))

            groups[group_id].extend([i for d, i in nbs])
            group_id += 1
        print(groups)

        # TODO: match ids back to hashtags and merge them back onto df.
        # TODO: check hashtags that only contain 1 rel_id, if it doesn't have
        # a partner, then prune it.


def cosine_similarities(df, strings, sim_thresh=0.8, max_id=0,
                        output_col='text_id', n_neighbors=100):
    _out('creating tf-idf matrix...')
    tf_idf_matrix = _tf_idf(strings, analyzer=_ngrams)

    _out('computing cosine similarities...')
    cos_sim = cosine_similarity(tf_idf_matrix)

    _out('filtering out simiarities below threshold...')
    cos_sim[cos_sim < sim_thresh] = 0.0

    _out('converting matrix to sparse matrix...')
    scm = sparse.csr_matrix(cos_sim)

    _out('putting matches into groups...')
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


# private
def _ngrams(string, n=3):
    string = re.sub(r'[,-./]|\sBD', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def _out(message=''):
    sys.stdout.write(message + '\n')
    sys.stdout.flush()


def _split_data(df, approx_datapoints=120000, in_col='text'):
    delta = 100000000

    if len(df.groupby(in_col).size()) <= approx_datapoints:
        _out('found optimal num pieces: 1')
        return [df]

    for i in range(2, 100):
        dps = []
        pieces = np.array_split(df, i)

        for piece in pieces:
            dps.append(len(piece.groupby(in_col).size()))

        mean_dps = np.mean(dps)
        _out('num pieces: %d, mean datapoints: %.2f' % (i, mean_dps))

        new_delta = np.abs(approx_datapoints - mean_dps)
        if new_delta < delta:
            delta = new_delta
        else:
            _out('found optimal num pieces: %d' % (i - 1))
            pieces = np.array_split(df, i - 1)
            return pieces


def _tf_idf(strings, analyzer='word'):
    vectorizer = TfidfVectorizer(min_df=1, analyzer=analyzer)
    tf_idf_matrix = vectorizer.fit_transform(strings)
    return tf_idf_matrix


if __name__ == '__main__':
    domain = 'twitter'
    info_type = 'hashtag'
    in_dir = 'independent/data/' + domain + '/'
    df = pd.read_csv(in_dir + info_type + '.csv', nrows=1000)
    print(df)
    # similarities(df, num_chunks=1, target_col=info_type,
    #              out_col=info_type + '_id')
    knn_similarities(df, in_col=info_type)
