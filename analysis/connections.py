"""
Module to find subnetwork of a data points based on its relationships.
"""
from collections import Counter


class Connections:

    def __init__(self):
        self.size_threshold = 100

    # public
    def consolidate(self, subgraphs, max_size=40000):
        """Combine subgraphs into larger sets to reduce total number of
        subgraphs to do inference over."""
        sgs = []

        new_ids, new_rels = set(), set()
        for ids, rels in subgraphs:
            if len(new_ids) + len(ids) < max_size:
                new_ids.update(ids)
                new_rels.update(rels)
            elif len(new_ids) == 0 and len(ids) > max_size:
                new_ids.update(ids)
                new_rels.update(rels)
            else:
                sgs.append((new_ids, new_rels))
                new_ids, new_rels = ids, rels

        if len(new_ids) > 0:
            sgs.append((new_ids, new_rels))

        total = 0
        for ids, rels in sgs:
            total += len(ids)

        print('\nnumber of subgraphs: %d' % len(sgs))
        print('\nnumber of msgs: %d' % total)

        return sgs

    def find_subgraphs(self, df, relations):
        df = df.copy()
        all_ids = set(df['com_id'])
        subnets = []

        while len(all_ids) > 0:
            remain_df = df[df['com_id'].isin(all_ids)]
            com_id = all_ids.pop()
            subnet = self.subnetwork(com_id, remain_df, relations)
            subnets.append(subnet)
            [all_ids.discard(c_id) for c_id in subnet[0]]

        subgraphs = self._aggregate_single_node_subgraphs(subnets)
        self._validate_subgraphs(subgraphs)
        subgraphs = sorted(subgraphs, key=lambda x: len(x[0]))

        total = 0
        for ids, rels in subgraphs:
            total += len(ids)

        print('\nnumber of subgraphs: %d' % len(subgraphs))
        print('\nnumber of msgs: %d' % total)

        return subgraphs

    def subnetwork(self, com_id, df, relations, debug=False):
        """Public interface to find a subnetwork given a specified comment.
        com_id: identifier of target comment.
        df: comments dataframe.
        relations: list of relations as tuples.
        debug: boolean to print extra information.
        Returns set of comment ids in subnetwork, and relations used."""
        direct, rels = self.direct_connections(com_id, df, relations)

        if len(direct) < self.size_threshold:
            result = self.group(com_id, df, relations, debug=debug)
        else:
            result = self.iterate(com_id, df, relations, debug=debug)
        return result

    # private
    def _aggregate_single_node_subgraphs(self, subnets):
        no_rel_ids = [s.pop() for s, r in subnets if r == set()]
        no_rel_sg = (no_rel_ids, set())
        rel_sgs = [s for s in subnets if s[1] != set()]

        subgraphs = rel_sgs.copy()
        subgraphs.append(no_rel_sg)
        return subgraphs

    def iterate(self, com_id, df, relations, debug=False):
        """Finds all comments directly and indirectly connected to com_id.
        com_id: identifier of target comment.
        df: comments dataframe.
        relations: list of relations as tuples.
        debug: boolean to print extra information.
        Returns set of comment ids in subnetwork, and relations used."""
        g_ids = [r[2] for r in relations]
        com_df = df[df['com_id'] == com_id]
        first_pass = True
        converged = False
        passes = 0

        # init group values as sets in a dict.
        g_dict, g_cnt = {}, {}
        for rel, g, g_id in relations:
            g_dict[g_id] = set(com_df[g_id].values)
            g_cnt[g_id] = 0

        total_cc = set()
        while first_pass or not converged:
            passes += 1
            cc = set()

            if debug:
                print('pass ' + str(passes))

            for ndx, row in df.iterrows():
                connected = False

                # method
                for g_id in g_ids:
                    if row[g_id] in g_dict[g_id]:
                        connected = True
                        cc.add(row['com_id'])
                        if row['com_id'] not in total_cc:
                            g_cnt[g_id] += 1
                        break

                # method
                if connected is True:
                    for g_id in g_ids:
                        g_dict[g_id].add(row[g_id])

            if first_pass:
                first_pass = False
                total_cc = cc
            else:
                if len(cc) != len(total_cc):
                    total_cc = cc
                else:
                    converged = True
        rels = set([r for r, g, gid in relations if g_cnt[gid] > 1])
        return total_cc, rels

    def group(self, target_id, df, relations, debug=False):
        """Finds all comments directly and indirectly connected to target_id.
        target_id: specified comment to find connections of.
        df: comments dataframe.
        relations: relations that might be present in the subnetwork.
        debug: if True, prints information about intermediate connections.
        Returns a set of comment ids in the subnetwork, a set of relations
                present in the subnetwork."""
        subnetwork = set({target_id})
        frontier, direct_connections = set({target_id}), set()
        relations_present = set()
        tier = 1

        while len(frontier) > 0:
            com_id = frontier.pop()
            connections, dir_rels = self.direct_connections(com_id, df,
                                                            relations)
            unexplored = [c for c in connections if c not in subnetwork]

            # switch to iteration method if subnetwork is too large.
            if len(connections) >= self.size_threshold:
                return self.iterate(target_id, df, relations, debug)

            # update sets.
            subnetwork.update(unexplored)
            direct_connections.update(unexplored)
            relations_present.update(dir_rels)

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

    def direct_connections(self, com_id, df, possible_relations):
        """Finds all data points associated with the specified comment.
        com_id: id of the comment in question.
        df: comments dataframe.
        possible_relations: relationships to use to find connections.
        Returns a set of com_ids that are connected to com_id, a set of
                relations active in these connections."""
        com_df = df[df['com_id'] == com_id]
        subnetwork, relations = set(), set()

        list_filter = lambda l, v: True if v in l else False

        for relation, group, group_id in possible_relations:
            g_vals = com_df[group_id].values

            if len(g_vals) > 0:
                vals = g_vals[0]

                for val in vals:
                    rel_df = df[df[group_id].apply(list_filter, args=(val,))]
                    if len(rel_df) > 1:
                        relations.add(relation)
                        subnetwork.update(set(rel_df['com_id']))
        return subnetwork, relations

    def _validate_subgraphs(self, subgraphs):
        id_list = []

        for ids, rels in subgraphs:
            id_list.extend(list(ids))

        for v in Counter(id_list).values():
            assert v == 1
