"""
This module handles all operations to build a Markov Network and run
Loopy Belief Propagation on that network.
"""
import os
import pandas as pd


class MRF:
    """Class that handles all operations pertaining to the MRF."""

    def __init__(self, config_obj, util_obj, generator_obj):
        """Initialize all object dependencies for this class."""

        self.config_obj = config_obj
        """User settings."""
        self.util_obj = util_obj
        """General utility methods."""
        self.generator_obj = generator_obj
        """Object to create group ids for relations."""

    # public
    def run(self, mrf_f):
        """Runs loopy bp on the resulting the MRF model using Libra.
        mrf_f: folder where the mn file is."""
        print('running bp...')
        cwd = os.getcwd()
        execute = 'libra bp -m model.mn -mo marginals.txt'
        os.chdir(mrf_f)  # change to mrf directory
        os.system(execute)
        os.chdir(cwd)  # change back to original directory

    def process_marginals(self, msgs_dict, mrf_f, pred_dir=''):
        fold = self.config_obj.fold
        preds = []

        with open(mrf_f + 'marginals.txt', 'r') as f:
            for i, line in enumerate(f.readlines()):
                for msg_id, msg_dict in msgs_dict.items():
                    if msg_dict['ndx'] == i:
                        pred = line.split(' ')[1]
                        preds.append((int(msg_id), float(pred)))

        df = pd.DataFrame(preds, columns=['com_id', 'mrf_pred'])
        df.to_csv(pred_dir + 'mrf_preds_' + fold + '.csv', index=None)

    def clear_data(self, data_f, fw=None):
        """Clears any old predicate or model data.
        data_f: folder where mrf data is stored.
        fw: file writer."""
        self.util_obj.write('clearing out old data...', fw=fw)
        os.system('rm ' + data_f + '*.mn')
        os.system('rm ' + data_f + '*.txt')

    def gen_mn(self, df, dset, rel_data_f, fw=None):
        """Generates markov network based on specified relationships.
        df: original dataframe.
        dset: dataset (e.g. 'val', 'test').
        rel_data_f: folder to save network to.
        fw: file writer."""
        relations = self.config_obj.relations
        rel_dicts = []

        # df = self.generator_obj.gen_group_ids(df, relations)
        msgs_dict, ndx = self._priors(df)
        for rel, group, group_id in relations:
            rel_df = self.generator_obj.gen_rel_df(df, rel)
            rel_dict, ndx = self._relation(rel_df, rel, group, group_id, ndx)
            rel_dicts.append((rel_dict, rel))
        self._print_network_size(msgs_dict, rel_dicts)
        self._write_model_file(msgs_dict, rel_dicts, ndx, rel_data_f)
        return msgs_dict

    # private
    def _priors(self, df, card=2):
        msgs_dict = {}

        msgs = list(df['com_id'])
        priors = list(df['ind_pred'])
        msg_priors = list(zip(msgs, priors))

        for i, (msg_id, prior) in enumerate(msg_priors):
            msgs_dict[msg_id] = {'ndx': i, 'prior': prior, 'card': card}

        ndx = len(msgs_dict)
        return msgs_dict, ndx

    # TODO: make this work with msgs in multiple hubs
    def _relation(self, rel_df, relation, group, group_id, ndx):
        rels_dict = {}
        print(relation, group, group_id)

        g = rel_df.groupby(group_id)
        for index, row in g:
            if len(row) > 1:
                rel_id = list(row[group_id])[0]
                msgs = list(row['com_id'])
                rels_dict[rel_id] = {'ndx': ndx, relation: msgs}
                ndx += 1
        # print(rels_dict)
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
                print(i, msg_dict)

            # write pairwise node factors
            for rel_dict, relation in rel_dicts:
                for group_id, group_dict in rel_dict.items():
                    rel_ndx = group_dict['ndx']
                    msg_ids = group_dict[relation]
                    print(rel_ndx, group_id, msg_ids)

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

    def _print_network_size(self, msgs_dict, rel_dicts):
        total_nodes, total_edges = len(msgs_dict), 0
        print('\nNetwork Size:')

        print('\tmsg nodes: %d' % (len(msgs_dict)))
        for rel_dict, relation in rel_dicts:
            total_nodes += len(rel_dict)
            edges = 0

            for group_id, group_dict in rel_dict.items():
                edges += len(group_dict[relation])

            t = (relation, len(rel_dict), edges)
            print('\t%s nodes: %d, edges: %d' % t)
            total_edges += edges

        print('All Nodes: %d, All Edges: %d\n' % (total_nodes, total_edges))
