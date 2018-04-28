"""
This module handles all operations to run the relational model using psl.
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict


class PSL:

    def __init__(self, config_obj, conns_obj, draw_obj, pred_builder_obj,
                 util_obj):
        self.config_obj = config_obj
        self.conns_obj = conns_obj
        self.draw_obj = draw_obj
        self.pred_builder_obj = pred_builder_obj
        self.util_obj = util_obj

    # public
    def clear_data(self, data_f):
        self.util_obj.out('clearing out old data...')
        os.system('rm ' + data_f + '*.tsv')
        os.system('rm ' + data_f + '*.txt')
        os.system('rm ' + data_f + 'db/*.db')

    def clear_preds(self, rel_d):
        self.util_obj.out('clearing out old preds...')
        path = rel_d + 'psl_preds_' + self.config_obj.fold + '.csv'
        os.system('rm -f %s' % path)

    def compile(self, psl_f):
        t1 = self.util_obj.out('compiling reasoning engine...')

        mvn_compile = 'mvn compile -q'
        mvn_build = 'mvn dependency:build-classpath '
        mvn_build += '-Dmdep.outputFile=classpath.out -q'

        self.util_obj.pushd(psl_f)
        os.system(mvn_compile)
        os.system(mvn_build)
        self.util_obj.popd()
        self.util_obj.time(t1)

    def infer(self, df, psl_d, psl_f, rel_d, max_size=500000):
        fold = self.config_obj.fold
        relations = self.config_obj.relations

        g, ccs = self.conns_obj.find_subgraphs(df, relations, max_size)
        # self._analyze_connected_components(ccs, df)
        # exit(0)
        subgraphs = self.conns_obj.consolidate(ccs, max_size)

        for i, (ids, hubs, rels, edges) in enumerate(subgraphs):
            _id = i + int(fold)
            sg_df = df[df['com_id'].isin(ids)]
            self._gen_predicates(sg_df, 'test', psl_d, _id)
            self._network_size(psl_d, _id)

            t1 = self.util_obj.out('reasoning over sg_%d...' % i)
            self._run(psl_f, _id)
            self.util_obj.time(t1)
        self._combine_predictions(len(subgraphs), rel_d)

        # self._analyze_connected_components(ccs, df)

        # if self.config_obj.has_display:
        #     preds_df = pd.read_csv(rel_d + 'psl_preds_' + fold + '.csv')
        #     new_df = df.merge(preds_df, how='left')
        #     self.draw_obj.draw_graphs(new_df, g, ccs, relations,
        #                               dir='graphs/', col='psl_pred')

    def train(self, df, psl_d, psl_f):
        self._gen_predicates(df, 'val', psl_d)
        self._gen_model(psl_d)
        self._network_size(psl_d)

        t1 = self.util_obj.out('training...')
        self._run(psl_f)
        self.util_obj.time(t1)

    # private
    def _combine_predictions(self, num_subgraphs, rel_d):
        fold = self.config_obj.fold
        dfs = []

        for i in range(num_subgraphs):
            s_id = str(i + int(fold))
            df = pd.read_csv(rel_d + 'psl_preds_' + s_id + '.csv')
            dfs.append(df)
        df = pd.concat(dfs)
        df.to_csv(rel_d + 'psl_preds_' + fold + '.csv', index=None)

    def _gen_model(self, data_f):
        rules = []

        rules.extend(self._priors())
        for relation, group, group_id in self.config_obj.relations:
            rules.extend(self._map_relation_to_rules(relation, group))
        self._write_model(rules, data_f)

    def _gen_predicates(self, df, dset, rel_data_f, iden=None):
        r_dfs = []
        s_iden = self.config_obj.fold if iden is None else str(iden)

        self.pred_builder_obj.build_comments(df, dset, rel_data_f, iden=s_iden)
        for relation, group, group_id in self.config_obj.relations:
            r_df = self.pred_builder_obj.build_relations(relation, group,
                                                         group_id,
                                                         df, dset, rel_data_f,
                                                         iden=s_iden)
            r_dfs.append(r_df)
        return r_dfs

    def _map_relation_to_rules(self, relation, group, wgt=1.0, sq=True):
        rule1 = str(wgt) + ': '
        rule2 = str(wgt) + ': '

        atom1 = relation + '(Com, ' + group.capitalize() + ')'
        atom2 = 'spmy' + group + '(' + group.capitalize() + ')'
        atom3 = 'spam(Com)'

        rule1 += atom1 + ' & ' + atom2 + ' -> ' + atom3
        rule2 += atom1 + ' & ' + atom3 + ' -> ' + atom2

        if sq:
            rule1 += ' ^2'
            rule2 += ' ^2'
        return [rule1, rule2]

    def _network_size(self, data_f, iden=None):
        s_iden = self.config_obj.fold if iden is None else str(iden)
        relations = self.config_obj.relations
        dset = 'test' if self.config_obj.infer else 'val'
        all_nodes, all_edges = 0, 0

        self.util_obj.out('%s network:' % dset)
        fn_m = data_f + dset + '_' + s_iden + '.tsv'
        msg_nodes = self.util_obj.file_len(fn_m)
        self.util_obj.out('-> msg nodes: %d' % msg_nodes)
        all_nodes += msg_nodes

        for relation, group, group_id in relations:
            fn_r = data_f + dset + '_' + relation + '_' + s_iden + '.tsv'
            fn_g = data_f + dset + '_' + group + '_' + s_iden + '.tsv'
            edges = self.util_obj.file_len(fn_r)
            hubs = self.util_obj.file_len(fn_g)
            t = (relation, hubs, edges)
            self.util_obj.out('-> %s nodes: %d, edges: %d' % t)

            all_edges += edges
            all_nodes += hubs

        t = (all_nodes, all_edges)
        self.util_obj.out('-> all nodes: %d, all edges: %d' % t)
        return all_edges

    def _run(self, psl_f, iden=None):
        s_iden = self.config_obj.fold if iden is None else str(iden)
        fold = self.config_obj.fold
        domain = self.config_obj.domain
        action = 'Infer' if self.config_obj.infer else 'Train'
        relations = [r[0] for r in self.config_obj.relations]

        arg_list = [fold, s_iden, domain] + relations
        execute = 'java -Xmx60g -cp ./target/classes:`cat classpath.out` '
        execute += 'spam.' + action + ' ' + ' '.join(arg_list)

        self.util_obj.pushd(psl_f)
        os.system(execute)
        self.util_obj.popd()

    def _priors(self, wgt=1.0, sq=True):
        neg_prior = str(wgt) + ': ~spam(Com)'
        pos_prior = str(wgt) + ': indpred(Com) -> spam(Com)'

        if sq:
            neg_prior += ' ^2'
            pos_prior += ' ^2'
        return [neg_prior, pos_prior]

    def _write_model(self, rules, data_f):
        fold = self.config_obj.fold

        with open(data_f + 'rules_' + fold + '.txt', 'w') as w:
            for rule in rules:
                w.write(rule + '\n')

    def _analyze_connected_components(self, ccs, df):
        self.util_obj.out('analyzing connected components...')

        d = {'cnt': defaultdict(int), 'same_cnt': defaultdict(int),
             'mean': defaultdict(list), 'median': defaultdict(list),
             'std': defaultdict(list)}
        keys = set()

        ccs = [x for x in ccs if x[3] > 0]  # filter out no edge subgraphs
        for msg_nodes, hub_nodes, relations, edges in ccs:
            print(msg_nodes)
            print(hub_nodes)
            print(relations)
            print(len(msg_nodes))
            print()

            qf = df[df['com_id'].isin(msg_nodes)]
            ip = qf['ind_pred']
            l = len(msg_nodes)
            keys.add(l)
            print(qf)

            d['cnt'][l] += 1
            if np.allclose(ip, ip[::-1], atol=1e-4):
                d['same_cnt'][l] += 1
            else:
                d['mean'][l].append(np.mean(ip))
                d['median'][l].append(np.median(ip))
                d['std'][l].append(np.std(ip))
        print(d)

        for k in keys:
            cnt = d['cnt'][k]
            sme_cnt = d['same_cnt'][k]
            mean = np.mean(d['mean'][k])
            median = np.mean(d['median'][k])
            std = np.mean(d['std'][k])

            t = (k, cnt, sme_cnt, mean, median, std)
            s = '%d:, cnt: %d, same_cnt: %d, '
            s += 'mean: %.2f, median: %.2f, std: %.2f'
            print(s % t)
