"""
This module handles all operations to build a Markov Network and run
Loopy Belief Propagation on that network.
"""
import os
import math
import pandas as pd
from operator import itemgetter
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

    def infer(self, df, ep, mrf_f, rel_pred_f, max_size=7500):
        md, rd = self._gen_mn(df, 'test', mrf_f, ep)
        size = self._network_size(md, rd, dset='test')

        if size > max_size:  # break the graph into subgraphs
            self.util_obj.out('size > %d...' % max_size)
            relations = self.config_obj.relations
            subgraphs = self.conns_obj.find_subgraphs(df, relations)
            subgraphs = self.conns_obj.consolidate(subgraphs, max_size)

            dfs = []
            for i, (ids, rels, edges) in enumerate(subgraphs):
                s = 'reasoning over sg_%d with %d msgs and %d edges...'
                t1 = self.util_obj.out(s % (i, len(ids), edges))
                sg_df = df[df['com_id'].isin(ids)]
                md, rd = self._gen_mn(sg_df, 'test', mrf_f, ep)
                self._run(mrf_f, dset='test')
                df = self._process_marginals(md, mrf_f, dset='test',
                                             pred_dir=rel_pred_f)
                dfs.append(df)
                self.util_obj.time(t1)
            df = pd.concat(dfs)
            fold = self.config_obj.fold
            df.to_csv(rel_pred_f + 'mrf_preds_' + fold + '.csv', index=None)

        else:  # do inference over the entire set
            t1 = self.util_obj.out('inferring...')
            self._run(mrf_f, dset='test')
            self._process_marginals(md, mrf_f, dset='test',
                                    pred_dir=rel_pred_f)
            self.util_obj.time(t1)

    def tune_epsilon(self, df, mrf_f, rel_pred_f,
                     epsilons=[0.1, 0.2, 0.3, 0.4]):
        md, rd = self._gen_mn(df, 'val', mrf_f, 0.1)
        self._network_size(md, rd, dset='val')

        ut = self.util_obj
        ut.out('tuning %s:' % str(epsilons))

        ep_scores = []
        for i, ep in enumerate(epsilons):
            t1 = ut.out('%.2f...' % ep)
            md, rd = self._gen_mn(df, 'val', mrf_f, ep)
            self._run(mrf_f, dset='val')
            preds_df = self._process_marginals(md, mrf_f, dset='val',
                                               pred_dir=rel_pred_f)
            ep_score = self._compute_aupr(preds_df, df)
            ep_scores.append((ep, ep_score))
            self.util_obj.time(t1)
        b_ep = max(ep_scores, key=itemgetter(1))[0]
        ut.out('-> best epsilon: %.2f' % b_ep)
        return b_ep

    # private
    def _compute_aupr(self, preds_df, val_df):
        df = preds_df.merge(val_df, on='com_id', how='left')
        aupr = average_precision_score(df['label'], df['mrf_pred'])
        return aupr

    def _gen_mn(self, df, dset, rel_data_f, epsilon=0.1):
        fname = dset + '_model.mn'
        relations = self.config_obj.relations
        rel_dicts = []

        msgs_dict, ndx = self._priors(df, transform='logit')
        for rel, group, group_id in relations:
            rel_df = self.gen_obj.rel_df_from_rel_ids(df, group_id)
            rel_dict, ndx = self._relation(rel_df, rel, group, group_id, ndx)
            rel_dicts.append((rel_dict, rel))
        self._write_model_file(msgs_dict, rel_dicts, ndx, rel_data_f,
                               epsilon=epsilon, fname=fname)
        return msgs_dict, rel_dicts

    def _network_size(self, msgs_dict, rel_dicts, dset='val'):
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

    def _priors(self, df, card=2, transform='logit'):
        msgs_dict = {}

        df = self._transform_priors(df, transform=transform)
        priors = list(df['ind_pred'])
        msgs = list(df['com_id'])
        priors = list(df['ind_pred'])
        msg_priors = list(zip(msgs, priors))

        for i, (msg_id, prior) in enumerate(msg_priors):
            msgs_dict[msg_id] = {'ndx': i, 'prior': prior, 'card': card}

        ndx = len(msgs_dict)
        return msgs_dict, ndx

    def _process_marginals(self, msgs_dict, mrf_f, dset='test', pred_dir=''):
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

    def _run(self, mrf_f, dset='test'):
        model_name = dset + '_model.mn'
        marginals_name = dset + '_marginals.txt'

        cwd = os.getcwd()
        execute = 'libra bp -m %s -mo %s' % (model_name, marginals_name)
        os.chdir(mrf_f)  # change to mrf directory
        os.system(execute)
        os.chdir(cwd)  # change back to original directory

    def _transform_priors(self, df, col='ind_pred', transform='logit'):
        clf = self.config_obj.classifier
        df = df.copy()

        if clf != 'lr':
            if transform is not None:
                if transform == 'e':
                    scale = self._transform_e
                elif transform == 'logit':
                    scale = self._transform_logit
                elif transform == 'logistic':
                    scale = self._transform_logistic

                df['ind_pred'] = df['ind_pred'].apply(scale)
        return df

    def _transform_e(self, x):
        result = x

        if x == 0:
            result = 0
        elif x == 1:
            result == 1
        else:
            result = math.exp(x)
        return result

    def _transform_logistic(self, x, alpha=2):
        result = x

        if x == 0:
            result = 0
        elif x == 1:
            result == 1
        else:
            result = (x ** alpha) / (x + ((1 - x) ** alpha))
        return result

    def _transform_logit(self, x):
        result = x

        if x == 0:
            result = 0
        elif x == 1:
            result == 1
        else:
            result = math.log(x / (1 - x))
        return result

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
