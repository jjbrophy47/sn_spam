"""
Module that creates graphical features from a list of follow actions.
"""
import os
import pandas as pd


class GraphFeatures:
    """Class that handles all operations for creating graphical features."""

    def __init__(self, config_obj, util_obj):
        """Initialize object dependencies."""

        self.config_obj = config_obj
        """User setttings."""
        self.util_obj = util_obj
        """General utility methods."""

    # public
    def build(self, train_df, val_df, test_df):
        """Builds or loads graphical features.
        train_df: training dataframe.
        val_df: validation dataframe.
        test_df: testing dataframe.
        Returns graph features dataframe and list."""
        domain = self.config_obj.domain
        feats_df, feature_list = None, []

        if domain == 'soundcloud' or domain == 'twitter':
            data_f, gl_f, feat_f = self.define_file_folders()

            if not self.check_for_file(feat_f, 'graph_features.csv'):
                coms_df = self.concat_coms(train_df, val_df, test_df)
                net_sg, feats_sf = self.load_graph(coms_df, data_f, gl_f,
                        feat_f)
                feats_sf = self.build_features(net_sg, feats_sf)
                self.write_features(feats_sf, feat_f)

            self.util_obj.start('loading graph features...')
            feats_df = pd.read_csv(feat_f + 'graph_features.csv')
            feature_list = feats_df.columns.tolist()
            feature_list.remove('user_id')
            self.util_obj.end()
        return feats_df, feature_list

    # private
    def define_file_folders(self):
        """Returns absolute path directories."""
        ind_dir = self.config_obj.ind_dir
        domain = self.config_obj.domain

        data_f = ind_dir + 'data/' + domain + '/'
        gl_f = ind_dir + 'data/' + domain + '/graphlab/'
        feat_f = ind_dir + 'output/' + domain + '/features/'
        if not os.path.exists(feat_f):
            os.makedirs(feat_f)
        if not os.path.exists(gl_f):
            os.makedirs(gl_f)
        return data_f, gl_f, feat_f

    def concat_coms(self, train_df, val_df, test_df):
        """Appends validation and testing dataframes to training.
        train_df: training dataframe.
        val_df: validation dataframe.
        test_df: testing dataframe.
        Returns concatenated dataframe."""
        coms_df = pd.concat([train_df, val_df, test_df])
        coms_df['text'] = coms_df['text'].fillna('')
        coms_df = coms_df.reset_index()
        coms_df = coms_df.drop(['index'], axis=1)
        return coms_df

    def check_for_file(self, folder, filename):
        """Checks for the existence of a file.
        folder: folder where the file is.
        filename: name of the file in folder.
        Returns True if the file exists, False otherwise."""
        return os.path.exists(folder + filename)
        # return os.path.exists(folder + 'graph_features.csv')

    # def check_for_network_file(self, gl_f):
    #     return os.path.exists(gl_f + 'network.sg')

    def read_network_data(self, data_f):
        """Reads the raw network data.
        data_f: folder where the network data is.
        Returns SGraph created from the network data."""
        import graphlab as gl
        domain = self.config_obj.domain

        if domain == 'soundcloud':
            network_df = pd.read_csv(data_f + 'affiliations.csv',
                    usecols=['contact_uid', 'follower_uid'])
            src, dst = 'follower_uid', 'contact_uid'
        else:
            network_df = pd.read_csv(data_f + 'network.tsv',
                    delim_whitespace=True, header=None, names=['p1', 'p2'])
            src, dst = 'p1', 'p2'

        network_sf = gl.SFrame(network_df)
        network_sg = gl.SGraph()
        network_sg = network_sg.add_edges(edges=network_sf, src_field=src,
                dst_field=dst)
        return network_sg

    def load_graph(self, df, data_f, gl_f, feat_f):
        """Loads an SGraph if one is already available. Otherwise creates one.
        df: comments dataframe.
        data_f: folder where the network data is.
        gl_f: graphlab folder.
        feat_f: feature folder.
        Returns SGraph and an SFrame where the features will be recorded."""
        import graphlab as gl
        sf = gl.SFrame(df)

        if not self.check_for_file(gl_f, 'network.sg'):
            network_sg = self.read_network_data(data_f)
            network_sg.save(gl_f + 'network.sg')
        else:
            network_sg = gl.load_graph(gl_f + 'network.sg')

        features_sf = gl.SFrame(sf['user_id'])
        features_sf = features_sf.unique()
        features_sf.rename({'X1': 'user_id'})
        features_sf['user_id'] = features_sf['user_id'].astype(int)

        return network_sg, features_sf

    def build_features(self, network_sg, features_sf):
        """Runs various graph algorithms on the SGraph.
        network_sg: SGraph created from the network data.
        features_sf: SFrame containing all the users in the graph.
        Returns SFrame with values for each algorithm for each user."""
        import graphlab as gl

        if 'pagerank' not in features_sf.column_names():
            print('Running PageRank...')
            data_m = gl.pagerank.create(network_sg, verbose=False)
            temp_sf = data_m['pagerank']
            temp_sf.rename({'__id': 'user_id'})

            features_sf = features_sf.join(temp_sf, on='user_id', how='left')
            features_sf = features_sf.fillna('pagerank', 0)
            features_sf.remove_column('delta')

        if 'triangle_count' not in features_sf.column_names():
            print('Running triangle count')
            data_m = gl.triangle_counting.create(network_sg, verbose=False)
            temp_sf = data_m['triangle_count']
            temp_sf.rename({'__id': 'user_id'})
            features_sf = features_sf.join(temp_sf, on='user_id', how='left')
            features_sf = features_sf.fillna('triangle_count', 0)

        if 'core_id' not in features_sf.column_names():
            print('Running k-core')
            data_m = gl.kcore.create(network_sg, verbose=False)
            temp_sf = data_m['core_id']
            temp_sf.rename({'__id': 'user_id'})
            features_sf = features_sf.join(temp_sf, on='user_id', how='left')
            features_sf = features_sf.fillna('core_id', 0)

        if 'out_degree' not in features_sf.column_names():
            print('Running out-degree')
            temp_sf = network_sg.edges.groupby('__src_id',
                                               {'out_degree':
                                                gl.aggregate.COUNT()})
            temp_sf.rename({'__src_id': 'user_id'})
            features_sf = features_sf.join(temp_sf, on='user_id', how='left')
            features_sf = features_sf.fillna('out_degree', 0)

        if 'in_degree' not in features_sf.column_names():
            print('Running in-degree')
            temp_sf = network_sg.edges.groupby('__dst_id', {'in_degree':
                                               gl.aggregate.COUNT()})
            temp_sf.rename({'__dst_id': 'user_id'})
            features_sf = features_sf.join(temp_sf, on='user_id', how='left')
            features_sf = features_sf.fillna('in_degree', 0)

        return features_sf

    def write_features(self, features_sf, feat_f):
        """Writes the graph features to a csv file.
        features_sf: SFrame with all the graph features for each user.
        feat_f: feature folder."""
        features_sf.save(feat_f + 'graph_features.csv', format='csv')
