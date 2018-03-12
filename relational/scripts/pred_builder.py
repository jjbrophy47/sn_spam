"""
Module to group comments of a certain relation together.
"""


class PredicateBuilder:

    def __init__(self, config_obj, comments_obj, gen_obj, util_obj):
        self.config_obj = config_obj
        self.comments_obj = comments_obj
        self.gen_obj = gen_obj
        self.util_obj = util_obj

    # public
    def build_comments(self, df, dset, data_f, tuffy=False):
        """convenience method to build predicate data for comments.
        df: comments dataframe.
        dset: dataset (e.g. val, test).
        data_f: folder to store the predicate data to.
        tuffy: boolean indicating if tuffy is the engine being used."""
        self.comments_obj.build(df, dset, data_f, tuffy=tuffy)

    def build_relations(self, relation, group, group_id, df, dset, data_f,
            tuffy=False, fw=None):
        """Builds the predicates for each relation (e.g. posts, text, etc.).
        group_id: column to group the comments by (e.g. text_id).
        relation: name of comment in relation (e.g. inText).
        group: name of relation (e.g. text).
        dset: dataset (e.g. 'val', 'test').
        df: comments dataframe.
        data_f: relational data folder.
        tuffy: boolean indicating if tuffy is the engine being used.
        fw: file writer."""
        ind_dir = self.config_obj.ind_dir
        domain = self.config_obj.domain
        data_dir = ind_dir + 'data/' + domain + '/'

        r_df = self.gen_obj.rel_df_from_rel_ids(df, group_id, data_dir)
        g_df = self.get_group_df(r_df, group_id)
        # self.util_obj.print_stats(df, r_df, relation, dset, fw=fw)

        if tuffy:
            self.write_tuffy_predicates(dset, r_df, relation, group_id, data_f)
        else:
            self.write_files(dset, r_df, g_df, relation, group, group_id,
                    data_f)

    # private
    def get_group_df(self, r_df, group_id):
        g_df = r_df.groupby(group_id).size().reset_index()
        g_df.columns = [group_id, 'size']
        return g_df

    def write_tuffy_predicates(self, dset, r_df, relation, group_id, data_f):
        """Writes predicate files to be read by the relational model.
        dset: validation or testing.
        r_df: dataframe with comments containing the relation.
        relation: comments in a relation (e.g. inText).
        group_id: column to group comments by (e.g. text_id).
        data_f: relational data folder."""
        rel = relation.capitalize()

        with open(data_f + dset + '_evidence.txt', 'a') as ev:
            ev.write('\n')
            for index, row in r_df.iterrows():
                com_id = str(int(row.com_id))
                g_id = str(row[group_id])
                ev.write(rel + '(' + com_id + ', ' + g_id + ')\n')

    def write_files(self, dset, r_df, g_df, relation, group, group_id, data_f):
        """Writes predicate files to be read by the relational model.
        dset: validation or testing.
        r_df: dataframe with comments containing the relation.
        g_df: grouped dataframe of relation.
        relation: comments in a relation (e.g. inText).
        group: name of relation (e.g. text).
        group_id: column to group comments by (e.g. text_id).
        data_f: relational data folder."""
        fold = self.config_obj.fold

        r_df.to_csv(data_f + dset + '_' + relation + '_' + fold + '.tsv',
                sep='\t', columns=['com_id', group_id], index=None,
                header=None)
        g_df.to_csv(data_f + dset + '_' + group + '_' + fold + '.tsv',
                sep='\t', columns=[group_id], index=None, header=None)
