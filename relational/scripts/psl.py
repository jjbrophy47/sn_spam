"""
This module handles all operations to run the relational model using psl.
"""
import os
import pandas as pd


class PSL:
    """Class that handles all operations pertaining to PSL."""

    def __init__(self, config_obj, pred_builder_obj, util_obj):
        self.config_obj = config_obj
        self.pred_builder_obj = pred_builder_obj
        self.util_obj = util_obj
        self.wgt = 1.0
        self.sq = True

    # public
    def combine_predictions(self, num_subgraphs, rel_d):
        """Combine predictions from all subgraphs."""
        fold = self.config_obj.fold
        dfs = []

        for i in range(num_subgraphs):
            df = pd.read_csv(rel_d + 'psl_preds_' + str(i) + '.csv')
            dfs.append(df)
        df = pd.concat(dfs)
        df.to_csv(rel_d + 'psl_preds_' + fold + '.csv', index=None)

    def compile(self, psl_f):
        """Compiles PSL with groovy scripts.
        psl_f: psl folder."""
        print('\nCompiling reasoning engine...')
        mvn_compile = 'mvn compile -q'
        mvn_build = 'mvn dependency:build-classpath '
        mvn_build += '-Dmdep.outputFile=classpath.out -q'

        # os.chdir(psl_f)  # change to psl directory
        self.util_obj.pushd(psl_f)
        os.system(mvn_compile)
        os.system(mvn_build)
        self.util_obj.popd()

    def run(self, psl_f, iden=None):
        """Runs the PSL model using Java.
        psl_f: psl folder."""
        s_iden = self.config_obj.fold if iden is None else str(iden)
        fold = self.config_obj.fold
        domain = self.config_obj.domain
        action = 'Infer' if self.config_obj.infer else 'Train'
        relations = [r[0] for r in self.config_obj.relations]

        arg_list = [fold, s_iden, domain] + relations
        execute = 'java -Xmx60g -cp ./target/classes:`cat classpath.out` '
        execute += 'spam.' + action + ' ' + ' '.join(arg_list)

        # os.chdir(psl_f)  # change to psl directory
        self.util_obj.pushd(psl_f)
        os.system(execute)
        self.util_obj.popd()

    def clear_data(self, data_f, fw=None):
        """Clears any old predicate or model data.
        data_f: folder where psl data is stored.
        fw: file writer."""
        self.util_obj.write('clearing out old data...', fw=fw)
        os.system('rm ' + data_f + '*.tsv')
        os.system('rm ' + data_f + '*.txt')
        os.system('rm ' + data_f + 'db/*.db')

    def gen_predicates(self, df, dset, rel_data_f, iden=None):
        """Generates all necessary predicates for the relational model.
        df: original validation dataframe.
        dset: dataset (e.g. 'val', 'test').
        rel_data_f: folder to save predicate data to.
        fw: file writer."""
        s_iden = self.config_obj.fold if iden is None else str(iden)

        self.pred_builder_obj.build_comments(df, dset, rel_data_f, iden=s_iden)
        for relation, group, group_id in self.config_obj.relations:
            self.pred_builder_obj.build_relations(relation, group, group_id,
                                                  df, dset, rel_data_f,
                                                  iden=s_iden)

    def gen_model(self, data_f):
        """Generates a text file with all the rules of the relational model.
        data_f: folder to store the model in."""
        rules = []

        rules.extend(self.priors())
        for relation, group, group_id in self.config_obj.relations:
            rules.extend(self.map_relation_to_rules(relation, group))
        self.write_model(rules, data_f)

    def network_size(self, data_f, iden=None):
        """Counts the number of constants for each predicate to find
                the size of the resulting graphical model.
        data_f: data folder.
        iden: identifier of the graph or subgraph."""
        s_iden = self.config_obj.fold if iden is None else str(iden)
        relations = self.config_obj.relations
        dset = 'test' if self.config_obj.infer else 'val'
        size = 0

        for relation, group, group_id in relations:
            fname_r = data_f + dset + '_' + relation + '_' + s_iden + '.tsv'
            size += self.util_obj.file_len(fname_r)
        self.util_obj.out('network size: %d' % size)
        return size

    # private
    def priors(self):
        """Constructs negative and positive priors for the model.
        Returns a list of both rules in string form."""
        neg_prior = str(self.wgt) + ': ~spam(Com)'
        pos_prior = str(self.wgt) + ': indpred(Com) -> spam(Com)'

        if self.sq:
            neg_prior += ' ^2'
            pos_prior += ' ^2'
        return [neg_prior, pos_prior]

    def map_relation_to_rules(self, relation, group):
        """Constructs relational rules for the model.
        relation: the link between multiple comments.
        group: general name of the relationship.
        Returns a list of rules in string form."""
        rule1 = str(self.wgt) + ': '
        rule2 = str(self.wgt) + ': '

        atom1 = relation + '(Com, ' + group.capitalize() + ')'
        atom2 = 'spammy' + group + '(' + group.capitalize() + ')'
        atom3 = 'spam(Com)'

        rule1 += atom1 + ' & ' + atom2 + ' -> ' + atom3
        rule2 += atom1 + ' & ' + atom3 + ' -> ' + atom2

        if self.sq:
            rule1 += ' ^2'
            rule2 += ' ^2'
        return [rule1, rule2]

    def write_model(self, rules, data_f):
        """Write rules to a text file.
        rules: list of rules in string form.
        data_f: folder to save model to."""
        fold = self.config_obj.fold

        with open(data_f + 'rules_' + fold + '.txt', 'w') as w:
            for rule in rules:
                w.write(rule + '\n')
