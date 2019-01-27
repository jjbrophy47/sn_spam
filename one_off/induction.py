import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from generator import Generator
from connections import Connections
from sklearn.metrics import roc_auc_score, average_precision_score

pd.set_option('display.width', 181)

gen = Generator()
con = Connections()

domain = 'soundcloud'
n_msgs = 5000000
rels = [('haspost', 'post', 'post_gid')]
rels += [('hastext', 'text', 'text_gid')]
rels += [('haslink', 'link', 'link_gid')]
# rels += [('hashash', 'hash', 'hash_gid')]

# qf = pd.read_csv('independent/data/%s/comments.csv' % domain, nrows=n_msgs)
# all_msgs = {str(s) for s in set(qf['com_id'])}

# qf = gen.gen_relational_ids(qf, rels)
# g = con.build_networkx_graph(qf, rels, with_label=False, add_msg_nodes=True)
# ccs_raw = nx.connected_components(g)

# data_range = [38504979, 42783310]


def collect_train_connections():

    train_pct = 0.7125

    youtube = [[0, 2000000],
               [492386, 2492386],
               [984772, 2984772],
               [1477158, 3477158],
               [1969544, 3969544],
               [2461930, 4461930],
               [2954316, 4954316],
               [3446702, 5446702],
               [3939088, 5939088],
               [4431474, 6431474]]

    twitter = [[0, 884598],
               [884598, 1769196],
               [1769196, 2653794],
               [2653794, 3538392],
               [3538392, 4422990],
               [4422990, 5307588],
               [5307588, 6192186],
               [6192186, 7076784],
               [7076784, 7961382],
               [7961382, 8845980]]

    soundcloud = [[0, 4278331],
                  [4278331, 8556662],
                  [8556662, 12834993],
                  [12834993, 17113324],
                  [17113324, 21391655],
                  [21391655, 25669986],
                  [25669986, 29948317],
                  [29948317, 34226648],
                  [34226648, 38504979],
                  [38504979, 42783310]]

    dfs = []

    for j, data_range in enumerate(soundcloud):
        print('reading data...')
        qf = pd.read_csv('independent/data/%s/comments.csv' % domain, nrows=data_range[1])
        qf = qf[data_range[0]:data_range[1]]
        qf_tr = qf[:int(len(qf) * train_pct)]
        qf_te = qf[int(len(qf) * train_pct):]

        print('getting training messages...')
        tr_all = {str(s) for s in set(qf_tr['com_id'])}

        print('getting test messages...')
        te_all = {str(s) for s in set(qf_te['com_id'])}
        te_ham = {str(s) for s in set(qf_te[qf_te['label'] == 0]['com_id'])}
        te_spam = {str(s) for s in set(qf_te[qf_te['label'] == 1]['com_id'])}

        qf = gen.gen_relational_ids(qf, rels)
        g = con.build_networkx_graph(qf, rels, with_label=False, add_msg_nodes=False)
        ccs_raw = nx.connected_components(g)

        test_ham = set()
        test_spam = set()

        for i, ccs in enumerate(ccs_raw):
            msgs = {x for x in ccs if '_' not in str(x)}
            if len(msgs.intersection(te_all)) > 0:
                if len(msgs.intersection(tr_all)) > 0:
                    test_ham.update(msgs.intersection(te_ham))
                    test_spam.update(msgs.intersection(te_spam))

        print('connections; ham: %d/%d, spam: %d/%d' % (len(test_ham), len(te_ham), len(test_spam), len(te_spam)))

        test_ham_spam = test_ham.union(test_spam)
        no_conns = te_all.difference(test_ham_spam)  # test instances with no connections to training instances

        print(len(test_ham_spam), len(no_conns))

        df = pd.DataFrame(list(no_conns), columns=['com_id'])
        df.to_csv('%s_%d.csv' % (domain, j), index=None)
        dfs.append(df)

    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    df.to_csv('%s_include.csv' % domain, index=None)


def evaluate_stuff():
    domain = 'soundcloud'
    ranges = [70000, 71000, 72000]
    pred_dir = 'sc1'
    preds = ['ind_pred', 'psl_pred', 'mrf_pred']
    lf = pd.read_csv('independent/data/%s/comments.csv' % domain)[['com_id', 'label']]
    for r in ranges:
        print(r)
        pf = pd.read_csv('%s/merged_%ds.csv' % (pred_dir, r))
        ff = pd.read_csv('%s_include.csv' % domain)
        df = lf.merge(pf).merge(ff)
        for pred in preds:
            aupr = average_precision_score(df['label'], df[pred])
            auroc = roc_auc_score(df['label'], df[pred])
            print('%s, aupr: %.3f, auroc: %.3f' % (pred.replace('_pred', ''), aupr, auroc))


def merge_stuff():
    ranges = [11000, 12000, 13000]
    for r in ranges:
        print(r)
        dfm, dfp, dfi = [], [], []
        for i in range(r, r+10):
            print('->%d' % i)
            dfm.append(pd.read_csv('mrf_preds_%d.csv' % i))
            dfp.append(pd.read_csv('psl_preds_%d.csv' % i))
            dfi.append(pd.read_csv('test_%d_preds.csv' % i))
        dfm, dfp, dfi = pd.concat(dfm), pd.concat(dfp), pd.concat(dfi)
        df = dfm.merge(dfp).merge(dfi)
        df.to_csv('merged_%ds.csv' % r, index=None)


def plot_stuff():
    ham_means = np.array([0.37, 0.33, 0.82]) * 100
    ham_std = np.array([0.02, 0.05, 0.01]) * 100
    spam_means = np.array([0.56, 0.67, 0.89]) * 100
    spam_std = np.array([0.03, 0.11, 0.13]) * 100

    fig, ax = plt.subplots(figsize=(3, 4))
    index = np.arange(3)
    bar_width = 0.35
    opacity = 0.8

    ax.bar(index, ham_means, bar_width, yerr=ham_std, alpha=opacity, color='g', label='ham')
    ax.bar(index + bar_width, spam_means, bar_width, yerr=spam_std, alpha=opacity, color='r', label='spam')

    fontsize = 18
    ax.set_ylabel('% of messages', fontsize=fontsize)
    ax.set_title('(b)', fontsize=fontsize)
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(['YouTube', 'Twitter', 'SoundCloud'], rotation=20, ha='right', fontsize=fontsize - 3)
    ax.set_yticklabels([int(x) for x in ax.get_yticks()], fontsize=fontsize - 2)
    ax.legend(loc='upper left', fontsize=fontsize - 6)
    fig.tight_layout()
    plt.savefig('train_conns.pdf', bbox_inches='tight', format='pdf')
    plt.show()


def plot_connected_components():
    yt = pd.read_csv('youtube_5000000.csv')
    tw = pd.read_csv('twitter_5000000.csv')
    sc = pd.read_csv('soundcloud_5000000.csv')

    fig, ax = plt.subplots(figsize=(5, 4))
    fontsize = 18

    ax.plot(sc['connected_components'], sc['percentage'] * 100, 'orange', label='SoundCloud')
    ax.plot(yt['connected_components'], yt['percentage'] * 100, 'r:', label='YouTube')
    ax.plot(tw['connected_components'], tw['percentage'] * 100, 'b--', label='Twitter')

    ax.set_title('(a)', fontsize=fontsize + 1)
    ax.set_xlabel('# of connected components', fontsize=fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize - 1)
    ax.set_ylabel('% of data', fontsize=fontsize)
    ax.set_yticklabels([int(x) for x in ax.get_yticks()], fontsize=fontsize - 1)

    ax.set_xscale('log')
    ax.grid(b=True, which='both', axis='both')
    ax.legend()

    fig.tight_layout()
    plt.savefig('components.pdf', bbox_inches='tight', format='pdf')
    plt.show()


def connected_components_vs_data_coverage(qf, ccs_raw):
    ccs_list = []
    all_msgs = len(qf)

    for ccs in ccs_raw:
        ccs_msgs = {x for x in ccs if '_' not in str(x)}
        ccs_list.append(ccs_msgs)

    ccs_list.sort(key=len, reverse=True)
    incremental = set()
    x, y = [0], [0.0]

    for i, msgs in enumerate(ccs_list):
        incremental.update(msgs)
        percentage = len(incremental) / len(all_msgs)
        x.append(i + 1)
        y.append(percentage)

    df = pd.DataFrame(list(zip(x, y)), columns=['connected_components', 'percentage'])
    df.to_csv('%s_%d.csv' % (domain, len(qf)), index=None)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel('# of connected components')
    ax.set_xscale('log')
    ax.set_ylabel('% of data')
    ax.set_title('%s: %d messages' % (domain, len(qf)))
    ax.grid(b=True, which='both', axis='both')
    plt.show()


def filter_connected_components(ccs, spam_msgs, min_spam=0):
    spam_nodes = set()
    spam_sizes = []

    for cc in ccs:
        com_id_nodes = {x for x in cc if '_' not in str(x)}
        spam_in_cc = com_id_nodes.intersection(spam_msgs)
        if len(spam_in_cc) >= min_spam:
            spam_nodes.update(spam_in_cc)
            spam_sizes.append(len(com_id_nodes))

    return spam_nodes, spam_sizes
