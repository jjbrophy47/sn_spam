"""
This module handles all operations to run the relational model using psl.
"""
import os
import pandas as pd


class PSL:

    def __init__(self, config_obj, pred_builder_obj, util_obj):
        self.config_obj = config_obj
        self.pred_builder_obj = pred_builder_obj
        self.util_obj = util_obj

    # public
    def clear_data(self, data_f, fw=None):
        """Clears any old predicate or model data.
        data_f: folder where psl data is stored.
        fw: file writer."""
        self.util_obj.write('clearing out old data...', fw=fw)
        os.system('rm ' + data_f + '*.tsv')
        os.system('rm ' + data_f + '*.txt')
        os.system('rm ' + data_f + 'db/*.db')

    def compile(self, psl_f):
        """Compiles PSL with groovy scripts.
        psl_f: psl folder."""
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

        self._gen_predicates(df, 'test', psl_d)
        size = self._network_size(psl_d)

        if size >= max_size:  # do inference over subgraphs
            self.util_obj.out('size > %d...' % max_size)
            relations = self.config_obj.relations
            subgraphs = self.conns_obj.find_subgraphs(df, relations)
            subgraphs = self.conns_obj.consolidate(subgraphs, max_size)

            for i, (ids, rels, edges) in enumerate(subgraphs):
                _id = i + int(fold)
                s = 'reasoning over sg_%d with %d msgs and %d edges...'
                t1 = self.util_obj.out(s % (i, len(ids), edges))
                sg_df = df[df['com_id'].isin(ids)]
                self._gen_predicates(sg_df, 'test', psl_d, _id)
                self._run(psl_f, _id)
                self.util_obj.time(t1)
            self._combine_predictions(len(subgraphs), rel_d)
        else:
            t1 = self.util_obj.out('inferring...')
            self._run(psl_f)
            self.util_obj.time(t1)

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
        s_iden = self.config_obj.fold if iden is None else str(iden)

        self.pred_builder_obj.build_comments(df, dset, rel_data_f, iden=s_iden)
        for relation, group, group_id in self.config_obj.relations:
            self.pred_builder_obj.build_relations(relation, group, group_id,
                                                  df, dset, rel_data_f,
                                                  iden=s_iden)

    def _map_relation_to_rules(self, relation, group, wgt=1.0, sq=True):
        rule1 = str(wgt) + ': '
        rule2 = str(wgt) + ': '

        atom1 = relation + '(Com, ' + group.capitalize() + ')'
        atom2 = 'spammy' + group + '(' + group.capitalize() + ')'
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

        self.util_obj.out('\n%s network:' % dset)
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
