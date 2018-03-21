import re
import sys
import time
import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# public
def knn_similarities(df, sim_thresh=0.8, n_neighbors=100,
                     approx_datapoints=120000, max_feats=None,
                     in_col='text', out_col='text_id', out_dir='',
                     fname='sim.csv'):
    _out('splitting data into manageable chunks...')
    dfs = _split_data(df, approx_datapoints=approx_datapoints, in_col=in_col)
    all_ids = defaultdict(set)
    group_id = 0

    for n, chunk_df in enumerate(dfs):
        _out('creating tf-idf matrix for chunk %d...' % n)
        groups = defaultdict(lambda: set())
        g_df = chunk_df.groupby(in_col).size().reset_index()
        strings = list(g_df[in_col])
        tf_idf_matrix = _tf_idf(strings, analyzer=_ngrams, max_feats=max_feats)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(tf_idf_matrix)
        _out(str(tf_idf_matrix.shape))

        _out('querying/filtering each object for its closest neighbors...')
        for row in range(len(strings)):

            # if row % 100 == 0:
            #     _out('%d' % row)

            distances, indexes = nbrs.kneighbors(tf_idf_matrix.getrow(row))
            nbs = list(zip(distances[0], indexes[0]))
            nbs = [(d, i) for d, i in nbs if d <= sim_thresh]

            # _out('\n%s' % strings[row])
            # for d, i in nbs[:5]:
            #     _out('[%d] %s: %f' % (i, strings[i], d))

            groups[group_id].update(set([i for d, i in nbs]))
            group_id += 1

        groups = _merge_identical_groups(groups)
        ids = _assign_ids_to_items(groups, strings)
        all_ids = _aggregate_identical_keys(all_ids, ids)

    all_ids = _prune_single_items(all_ids, df, in_col)
    all_ids = _prune_redundant_ids(all_ids)
    print(all_ids)
    sim_df = _ids_to_dataframe(all_ids, df, in_col=in_col, out_col=out_col)
    sim_df.to_csv(out_dir + fname, index=None)


def cosine_similarities(df, sim_thresh=0.8, in_col='text',
                        out_col='text_id', approx_datapoints=120000,
                        max_feats=None, k=5, out_dir='', fname='sim.csv'):
    group_id = 0
    all_ids = defaultdict(set)
    dfs = _split_data(df, approx_datapoints=approx_datapoints, in_col=in_col)

    for n, chunk_df in enumerate(dfs):
        t1 = time.time()

        _out('creating tf-idf matrix for chunk %d...' % (n + 1))
        groups = defaultdict(set)
        g_df = chunk_df.groupby(in_col).size().reset_index()
        strings = list(g_df[in_col])
        tf_idf_matrix = _tf_idf(strings, analyzer=_ngrams, max_feats=max_feats)

        _out('computing cosine similarities...')
        cos_sim = cosine_similarity(tf_idf_matrix, dense_output=False)

        _out('filtering out simiarities below threshold...')
        scm = cos_sim >= sim_thresh

        _out('putting matches into groups...')
        for ndx in range(len(strings)):
            data = cos_sim[ndx].data
            indices = list(cos_sim[ndx].indices)
            sims = [(x, data[indices.index(x)]) for x in scm[ndx].indices]
            sims = sorted(sims, key=lambda x: x[1], reverse=True)
            sim_ids = [sim_ndx for sim_ndx, sim_val in sims[:k]]
            groups[group_id].update(set(sim_ids))
            group_id += 1

        _out('merge identical groups...')
        groups = _merge_identical_groups(groups)

        _out('assign ids to items...')
        ids = _assign_ids_to_items(groups, strings)

        _out('aggregate_identical_keys...')
        all_ids = _aggregate_identical_keys(all_ids, ids)

        _out('chunk time: %.4fm' % ((time.time() - t1) / 60.0))

    t1 = time.time()
    _out('\nprune single items...')
    all_ids = _prune_single_items(all_ids, df, in_col)
    _out('prune time: %.4fm' % ((time.time() - t1) / 60.0))

    t1 = time.time()
    _out('prune redundant ids...')
    all_ids = _prune_redundant_ids(all_ids)
    _out('prune time: %.4fm' % ((time.time() - t1) / 60.0))

    t1 = time.time()
    _out('put ids into a dataframe...')
    sim_df = _ids_to_dataframe(all_ids, df, in_col=in_col, out_col=out_col)
    _out('write to csv...')
    sim_df.to_csv(out_dir + fname, index=None)
    _out('time: %.4fm' % ((time.time() - t1) / 60.0))


# private
def _aggregate_identical_keys(all_ids, ids):
    for h1, group_ids in ids.items():
        all_ids[h1].update(group_ids)
    return all_ids


def _assign_ids_to_items(groups, strings):
    ids = defaultdict(set)
    for k, vals in groups.items():
        for v in vals:
            ids[strings[v]].add(k)
    return ids


def _ids_to_dataframe(all_ids, df, in_col='text', out_col='text_id'):
    rows = []
    for key, ids in all_ids.items():
        rows.extend([(key, v) for v in ids])

    sim_df = pd.DataFrame(rows, columns=[in_col, out_col])
    sim_df = df.merge(sim_df, on=in_col)
    sim_df = sim_df[['com_id', out_col]]
    return sim_df


def _merge_identical_groups(groups):
    g = {}
    vals = list(groups.values())

    while len(vals) > 0:
        v1 = vals.pop()
        keys = set()

        for k2, v2 in groups.items():
            if v1 == v2:
                keys.add(k2)

        vals = [v for v in vals if v != v1]
        g[min(keys)] = v1

    return g


def _ngrams(string, n=3):
    string = re.sub(r'[,-./]|\sBD', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def _out(message=''):
    sys.stdout.write(message + '\n')
    sys.stdout.flush()


def _prune_redundant_ids(all_ids):
    t1 = time.time()
    result = all_ids.copy()
    _out('\ncopying time: %.4fm' % ((time.time() - t1) / 60.0))

    l = [list(x) for x in list(all_ids.values())]
    ll = [x for sublist in l for x in sublist]
    group_ids = Counter(ll)

    _out('all_ids keys: %d, values: %d' % (len(all_ids.keys()), len(ll)))

    t1 = time.time()
    for i, (key, vals) in enumerate(all_ids.items()):
        if i % 1000 == 0:
            _out('%d ids pruned: %.4fm' % (i, ((time.time() - t1) / 60.0)))

        if len(vals) > 1:
            redundant_ids = set([v for v in vals if group_ids[v] == 1])

            if len(redundant_ids) > 1:
                redundant_ids.remove(min(redundant_ids))
                for redundant_id in redundant_ids:
                    result[key].remove(redundant_id)

    return result


def _prune_single_items(all_ids, df, in_col):
    all_ids = all_ids.copy()
    g_df = df.groupby(in_col).size().reset_index()
    g1 = list(g_df[g_df[0] == 1][in_col])

    for key in g1:
        if len(all_ids[key]) == 0 or len(all_ids[key]) == 1:
            all_ids.pop(key)

    return all_ids


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


def _tf_idf(strings, analyzer='word', max_feats=None):
    vectorizer = TfidfVectorizer(min_df=1, analyzer=analyzer,
                                 max_features=max_feats)
    tf_idf_matrix = vectorizer.fit_transform(strings)
    return tf_idf_matrix


if __name__ == '__main__':
    domain = 'twitter'
    info_type = 'hashtag'
    in_dir = 'independent/data/' + domain + '/'
    df = pd.read_csv(in_dir + info_type + '.csv')

    cosine_similarities(df, in_col=info_type, out_col=info_type + '_id',
                        out_dir=in_dir, fname=info_type + '_sim.csv',
                        max_feats=None, approx_datapoints=120000,
                        sim_thresh=0.7)
    # knn_similarities(df, in_col=info_type, out_col=info_type + '_id',
    #                  out_dir=in_dir, fname=info_type + '_sim.csv',
    #                  max_feats=None, approx_datapoints=120000)
