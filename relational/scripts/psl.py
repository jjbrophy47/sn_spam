"""
This module handles all operations to run the relational model using psl.
"""
import os


class PSL:
    """Class that handles all operations pertaining to PSL."""

    def __init__(self, config_obj, pred_builder_obj):
        """Initialize all object dependencies for this class."""

        self.config_obj = config_obj
        """User settings."""
        self.pred_builder_obj = pred_builder_obj
        """Builds predicate data."""
        self.wgt = 1.0
        """Initial weight for rules."""
        self.sq = True
        """Squared hinge loss if True, linear loss if False."""

    # public
    def run(self, psl_f):
        """Runs the PSL model using Java.
        psl_f: psl folder."""
        fold = self.config_obj.fold
        domain = self.config_obj.domain
        relations = [r[0] for r in self.config_obj.relations]

        print('Compiling relational model...')
        mvn_compile = 'mvn compile -q'
        mvn_build = 'mvn dependency:build-classpath '
        mvn_build += '-Dmdep.outputFile=classpath.out -q'
        arg_list = [fold, domain] + relations
        execute = 'java -Xmx60g -cp ./target/classes:`cat classpath.out` '
        execute += 'spam.Basic ' + ' '.join(arg_list)

        os.chdir(psl_f)  # change to psl directory
        os.system(mvn_compile)
        os.system(mvn_build)
        os.system(execute)

    def clear_data(self, data_f):
        """Clears any old predicate or model data.
        data_f: folder where psl data is stored."""
        print('Clearing out old data...')
        os.system('rm ' + data_f + '*.tsv')
        os.system('rm ' + data_f + '*.txt')
        os.system('rm ' + data_f + 'db/*.db')

    def gen_predicates(self, df, dset, rel_data_f):
        """Generates all necessary predicates for the relational model.
        df: original validation dataframe.
        dset: dataset (e.g. 'val', 'test').
        rel_data_f: folder to save predicate data to."""
        self.pred_builder_obj.build_comments(df, dset, rel_data_f)
        for relation, group, group_id in self.config_obj.relations:
            self.pred_builder_obj.build_relations(relation, group, group_id,
                    df, dset, rel_data_f)

    def gen_model(self, data_f):
        """Generates a text file with all the rules of the relational model.
        data_f: folder to store the model in."""
        rules = []

        rules.extend(self.priors())
        for relation, group, group_id in self.config_obj.relations:
            rules.extend(self.map_relation_to_rules(relation, group))
        self.write_model(rules, data_f)

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
