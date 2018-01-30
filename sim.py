import re
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def ngrams(string, n=3):
    string = re.sub(r'[,-./]|\sBD', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def tf_idf(strings, analyzer='word'):
    vectorizer = TfidfVectorizer(min_df=1, analyzer=analyzer)
    tf_idf_matrix = vectorizer.fit_transform(strings)
    return tf_idf_matrix

if __name__ == '__main__':
    print('extracting messages...')
    df = pd.read_csv('comments.csv', nrows=None)
    strings = list(df['text'])

    print('creating tf-idf matrix...')
    tf_idf_matrix = tf_idf(strings, analyzer=ngrams)

    print('computing cosine similarities...')
    cos_sim = cosine_similarity(tf_idf_matrix)

    print('filtering out simiarities below threshold...')
    cos_sim[cos_sim < 0.8] = 0.0

    print('converting matrix to sparse matrix...')
    scm = sparse.csr_matrix(cos_sim)

    print('putting matches into groups...')
    groups = {-1: set()}
    explored = set()
    i = 1

    for ndx in range(len(strings)):
        explored.add(ndx)
        matches = set(scm[ndx].indices)
        group = set(matches)

        if len(group) > 1:
            explored.update(matches)
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
    temp_df = pd.DataFrame(r, columns=['com_id', 'text_id'], index=indices)
    temp_df = temp_df.sort_index()
    temp_df = temp_df[temp_df['text_id'] != -1]
    temp_df.to_csv('sim.csv', index=None)
