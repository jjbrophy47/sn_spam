"""
Tests the graph_features module.
"""
import os
import unittest
import pandas as pd
import mock
from .context import graph_features
from .context import config
from .context import util
from .context import test_utils as tu


class GraphFeaturesTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        util_obj = util.Util()
        self.test_obj = graph_features.GraphFeatures(config_obj, util_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        result = self.test_obj

        self.assertTrue(isinstance(result.config_obj, config.Config))

    def test_build_wrong_domain(self):
        self.test_obj.config_obj.domain = 'youtube'

        result = self.test_obj.build('tr', 'va', 'te')

        self.assertTrue(result == (None, []))

    def test_build_correct_domain_with_graph_fetures(self):
        feats_df = tu.sample_df(10)
        feats_df.columns = ['user_id', 'random']
        self.test_obj.define_file_folders = mock.Mock(return_value=('data/',
                'gl/', 'feat/'))
        self.test_obj.check_for_file = mock.Mock(return_value=True)
        pd.read_csv = mock.Mock(return_value=feats_df)

        result = self.test_obj.build('tr_df', 'va_df', 'te_df')

        self.test_obj.define_file_folders.assert_called()
        self.test_obj.check_for_file.assert_called_with('feat/',
                'graph_features.csv')
        pd.read_csv.assert_called_with('feat/graph_features.csv')
        self.assertTrue(list(result[0]) == ['user_id', 'random'])
        self.assertTrue(len(result[0]) == 10)
        self.assertTrue(result[1] == ['random'])

    def test_build_correct_domain_without_graph_fetures(self):
        feats_df = tu.sample_df(10)
        feats_df.columns = ['user_id', 'random']
        self.test_obj.define_file_folders = mock.Mock(return_value=('data/',
                'gl/', 'feat/'))
        self.test_obj.check_for_file = mock.Mock(return_value=False)
        self.test_obj.concat_coms = mock.Mock(return_value='coms_df')
        self.test_obj.load_graph = mock.Mock(return_value=('sg', 'sf'))
        self.test_obj.build_features = mock.Mock(return_value=('sf2'))
        self.test_obj.write_features = mock.Mock()
        pd.read_csv = mock.Mock(return_value=feats_df)

        result = self.test_obj.build('tr', 'va', 'te')

        self.test_obj.define_file_folders.assert_called()
        self.test_obj.check_for_file.assert_called_with('feat/',
                'graph_features.csv')
        self.test_obj.concat_coms.assert_called_with('tr', 'va', 'te')
        self.test_obj.load_graph.assert_called_with('coms_df', 'data/',
                'gl/', 'feat/')
        self.test_obj.build_features.assert_called_with('sg', 'sf')
        self.test_obj.write_features.assert_called_with('sf2', 'feat/')
        pd.read_csv.assert_called_with('feat/graph_features.csv')
        self.assertTrue(list(result[0]) == ['user_id', 'random'])
        self.assertTrue(len(result[0]) == 10)
        self.assertTrue(result[1] == ['random'])

    def test_define_file_folders(self):
        os.makedirs = mock.Mock()

        result = self.test_obj.define_file_folders()

        self.assertTrue(result[0] == 'ind/data/soundcloud/')
        self.assertTrue(result[1] == 'ind/data/soundcloud/graphlab/')
        self.assertTrue(result[2] == 'ind/output/soundcloud/features/')

    def test_concat_coms(self):
        train = ['a', 'b', None]
        val = ['a', 'b', None]
        test = ['a', 'b', None]
        train = pd.DataFrame({'text': train})
        val = pd.DataFrame({'text': val})
        test = pd.DataFrame({'text': test})

        result = self.test_obj.concat_coms(train, val, test)

        expected = pd.Series(['a', 'b', '', 'a', 'b', '', 'a', 'b', ''])
        self.assertTrue(len(result) == 9)
        self.assertTrue(result['text'].equals(expected))

    def test_check_for_file(self):
        os.path.exists = mock.Mock(return_value=True)

        result = self.test_obj.check_for_file('feat/', 'file.csv')

        os.path.exists.assert_called_with('feat/file.csv')
        self.assertTrue(result)

    def _read_network_data_soundcloud(self):
        import graphlab as gl
        df = tu.sample_df(10)
        sf = gl.SFrame()
        sg = gl.SGraph()
        sg2 = gl.SGraph()
        sg.add_edges = mock.Mock(return_value=sg2)
        pd.read_csv = mock.Mock(return_value=df)
        gl.SFrame = mock.Mock(return_value=sf)
        gl.SGraph = mock.Mock(return_value=sg)

        result = self.test_obj.read_network_data('data/')

        pd.read_csv.assert_called_with('data/affiliations.csv',
                usecols=['contact_uid', 'follower_uid'])
        gl.SFrame.assert_called_with(df)
        gl.SGraph.assert_called()
        sg.add_edges.assert_called_with(edges=sf, src_field='follower_uid',
                dst_field='contact_uid')
        self.assertTrue(result == sg2)

    def _load_graph_exists(self):
        import graphlab as gl
        df = tu.sample_df(10)
        sf = gl.SFrame(df)
        sf.unique = mock.Mock(return_value=sf)
        sf.rename = mock.Mock()
        sg = gl.SGraph()
        gl.SFrame = mock.Mock(return_value=sf)
        gl.load_graph = mock.Mock(return_value=sg)
        self.test_obj.check_for_file(return_value=True)

        self.test_obj.load_graph('df', 'data/', 'gl/', 'feat/')

        gl.SFrame.call_args_list == [mock.call('df'), mock.call(sf['com_id'])]
        self.test_obj.check_for_file.assert_called_with('gl/', 'network.sg')
        gl.load_graph.assert_called_with('gl/network.sg')
        sf.unique.assert_called()
        sf.rename.assert_called_with({'X1': 'user_id'})

    def _build_features_pagerank(self):
        import graphlab as gl
        df = tu.sample_df(10)
        df['pagerank'] = 0.69
        sg = gl.SGraph()
        sf = gl.SFrame()
        data_m = gl.SFrame(df)
        sf.column_names = mock.Mock(return_value=['triangle_count', 'core_id',
                'out_degree', 'in_degree'])
        gl.pagerank.create = mock.Mock(return_value=data_m)
        sf.join = mock.Mock(return_value=sf)
        sf.fillna = mock.Mock(return_value=sf)
        sf.remove_column = mock.Mock()

        result = self.test_obj.build_features(self, sg, sf)

        self.assertTrue(result == sf)
        gl.pagerank.create.assert_called_with(sg, verbose=False)
        sf.join.assert_called()
        sf.fillna.assert_called_with('pagerank', 0)
        sf.remove_column.assert_called()

    def _write_features(self):
        import graphlab as gl
        sf = gl.SFrame()
        sf.save = mock.Mock()

        self.test_obj.write_features(sf, 'feat/')

        sf.save.assert_called_with('feat/graph_features.csv', format='csv')


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(
            GraphFeaturesTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
