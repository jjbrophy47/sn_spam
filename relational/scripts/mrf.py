"""
This module handles all operations to build a Markov Network and run
Loopy Belief Propagation on that network.
"""
import os
import math
import pandas as pd
from sklearn.metrics import average_precision_score


class MRF:

    def __init__(self, config_obj, util_obj, gen_obj):
        self.config_obj = config_obj
        self.util_obj = util_obj
        self.gen_obj = gen_obj

    # public
    def clear_data(self, data_f, fw=None):
        """Clears any old predicate or model data.
        data_f: folder where mrf data is stored.
        fw: file writer."""
        self.util_obj.write('clearing out old data...', fw=fw)
        os.system('rm ' + data_f + '*.mn')
        os.system('rm ' + data_f + '*.txt')

    def compute_aupr(self, preds_df, val_df):
        df = preds_df.merge(val_df, on='com_id', how='left')
        aupr = average_precision_score(df['label'], df['mrf_pred'])
        return aupr

    def gen_mn(self, df, dset, rel_data_f, epsilon=0.1):
        """Generates markov network based on specified relationships.
        df: original dataframe.
        dset: dataset (e.g. 'val', 'test').
        rel_data_f: folder to save network to."""
        fname = dset + '_model.mn'
        relations = self.config_obj.relations
        rel_dicts = []

        msgs_dict, ndx = self._priors(df, transform=None)
        for rel, group, group_id in relations:
            rel_df = self.gen_obj.rel_df_from_rel_ids(df, group_id)
            rel_dict, ndx = self._relation(rel_df, rel, group, group_id, ndx)
            rel_dicts.append((rel_dict, rel))
        self._write_model_file(msgs_dict, rel_dicts, ndx, rel_data_f,
                               epsilon=epsilon, fname=fname)
        return msgs_dict, rel_dicts

    def network_size(self, msgs_dict, rel_dicts, dset='val'):
        ut = self.util_obj
        total_nodes, total_edges = len(msgs_dict), 0
        ut.out('\n%s network:' % dset)

        ut.out('-> msg nodes: %d' % (len(msgs_dict)))
        for rel_dict, relation in rel_dicts:
            total_nodes += len(rel_dict)
            edges = 0

            for group_id, group_dict in rel_dict.items():
                edges += len(group_dict[relation])

            t = (relation, len(rel_dict), edges)
            ut.out('-> %s nodes: %d, edges: %d' % t)
            total_edges += edges

        ut.out('-> all nodes: %d, all edges: %d' % (total_nodes, total_edges))
        return total_edges

    def process_marginals(self, msgs_dict, mrf_f, dset='test', pred_dir=''):
        marginals_name = dset + '_marginals.txt'
        fold = self.config_obj.fold
        preds = []

        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        with open(mrf_f + marginals_name, 'r') as f:
            for i, line in enumerate(f.readlines()):
                for msg_id, msg_dict in msgs_dict.items():
                    if msg_dict['ndx'] == i:
                        pred = line.split(' ')[1]
                        preds.append((int(msg_id), float(pred)))

        df = pd.DataFrame(preds, columns=['com_id', 'mrf_pred'])
        df.to_csv(pred_dir + 'mrf_preds_' + fold + '.csv', index=None)
        return df

    def run(self, mrf_f, dset='test'):
        """Runs loopy bp on the resulting the MRF model using Libra.
        mrf_f: folder where the mn file is."""
        model_name = dset + '_model.mn'
        marginals_name = dset + '_marginals.txt'

        cwd = os.getcwd()
        execute = 'libra bp -m %s -mo %s' % (model_name, marginals_name)
        os.chdir(mrf_f)  # change to mrf directory
        os.system(execute)
        os.chdir(cwd)  # change back to original directory

    # private
    def _priors(self, df, card=2, transform=None):
        msgs_dict = {}

        priors = list(df['ind_pred'])

        if transform is not None:
            if transform == 'e':
                lambda x: math.exp(x)
            elif transform == 'logit':
                lambda x: math.log(x / 1 - x)
            elif transform == 'logistic':
                alpha = 2
                lambda x: (x ** alpha) / (x + ((1 - x) ** alpha))

        msgs = list(df['com_id'])
        priors = list(df['ind_pred'])
        msg_priors = list(zip(msgs, priors))

        for i, (msg_id, prior) in enumerate(msg_priors):
            msgs_dict[msg_id] = {'ndx': i, 'prior': prior, 'card': card}

        ndx = len(msgs_dict)
        return msgs_dict, ndx

    def _relation(self, rel_df, relation, group, group_id, ndx):
        rels_dict = {}

        g = rel_df.groupby(group_id)
        for index, row in g:
            if len(row) > 1:
                rel_id = list(row[group_id])[0]
                msgs = list(row['com_id'])
                rels_dict[rel_id] = {'ndx': ndx, relation: msgs}
                ndx += 1
        return rels_dict, ndx

    def _write_model_file(self, msgs_dict, rel_dicts, num_nodes, dir,
                          fname='model.mn', epsilon=0.15):
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(dir + fname, 'w') as f:
            # first line: comma separated cardinality for each node
            line = ''
            for i in range(num_nodes):
                line += '2\n' if i == num_nodes - 1 else '2,'
            f.write(line)

            # start network
            f.write('MN {\n')

            # write single node factors
            for i, (msg_id, msg_dict) in enumerate(msgs_dict.items()):
                assert i == msg_dict['ndx']
                prior = msg_dict['prior']
                ndx = msg_dict['ndx']
                factor = '%.5f +v%d_1\n'
                factor += '%.5f +v%d_0\n'
                f.write(factor % (prior, ndx, 1.0 - prior, ndx))

            # write pairwise node factors
            for rel_dict, relation in rel_dicts:
                for group_id, group_dict in rel_dict.items():
                    rel_ndx = group_dict['ndx']
                    msg_ids = group_dict[relation]

                    for msg_id in msg_ids:
                        msg_dict = msgs_dict[msg_id]
                        msg_ndx = msg_dict['ndx']

                        factor = '%.5f +v%d_0 +v%d_0\n'
                        factor += '%.5f +v%d_0 +v%d_1\n'
                        factor += '%.5f +v%d_1 +v%d_0\n'
                        factor += '%.5f +v%d_1 +v%d_1\n'

                        values = (1.0 - epsilon, msg_ndx, rel_ndx)
                        values += (epsilon, msg_ndx, rel_ndx)
                        values += (epsilon, msg_ndx, rel_ndx)
                        values += (1.0 - epsilon, msg_ndx, rel_ndx)

                        f.write(factor % values)
            f.write('}\n')
