"""
Module to find subnetwork of a data points based on its relationships.
"""


class Connections:
    """Class to find direct and indirect connections to a data point."""

    # public
    def all_connections(self, df, com_id, possible_relations, debug=False):
        """Show all comments connected to com_id and what relation they share.
        df: dataframe of comments.
        com_id: specified comment to find connections of.
        possible_relations: relations that might be present in the subnetwork.
        debug: if True, prints information about intermediate connections.
        Returns a set of comment ids in the subnetwork, a set of relations
                present in the subnetwork."""
        subnetwork = set({com_id})
        frontier, direct_connections = set({com_id}), set()
        tier = 1
        relations_present = set()

        while len(frontier) > 0:
            com_id = frontier.pop()
            connections, relations = self.direct_connections(df, com_id,
                    possible_relations)
            unexplored = [c for c in connections if c not in subnetwork]

            # update sets.
            subnetwork.update(unexplored)
            direct_connections.update(unexplored)
            relations_present.update(relations)

            # tier is exhausted, move to next level.
            if len(frontier) == 0 and len(direct_connections) > 0:
                id_df = df[df['com_id'].isin(list(direct_connections))]
                id_list = id_df['com_id'].values
                frontier = direct_connections.copy()
                direct_connections.clear()

                if debug:
                    print('\nTier ' + str(tier))
                    print(id_list, len(id_list))
                tier += 1

        return subnetwork, relations_present

    # private
    def direct_connections(self, df, com_id, possible_relations):
        """Finds all data points associated with the specified comment.
        df: comments dataframe.
        com_id: id of the comment in question.
        possible_relations: relationships to use to find connections.
        Returns a set of com_ids that are connected to com_id, a set of
                relations active in these connections."""
        com_df = df[df['com_id'] == com_id]
        subnetwork, relations = set(), set()

        for relation, group, group_id in possible_relations:
            cols = ['com_id', group_id]
            vals = com_df[group_id].values

            for val in vals:
                rel_df = df[df[group_id] == val]
                rel_df = rel_df.groupby(cols).size().reset_index()
                if len(rel_df) > 1:
                    relations.add(relation)
                    subnetwork.update(set(rel_df['com_id']))
        return subnetwork, relations
