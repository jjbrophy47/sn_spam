"""
Tests the connections module.
"""
import unittest
from .context import connections
from .context import test_utils as tu


class ConnectionsTestCase(unittest.TestCase):
    def setUp(self):
        self.test_obj = connections.Connections()

    def tearDown(self):
        self.test_obj = None

    def test_direct_connections_empty_sets(self):
        df = tu.sample_df(10)
        df.columns = ['com_id', 'user_id']
        rels = [('posts', 'user', 'user_id')]

        result = self.test_obj.direct_connections(df, 2, rels)

        self.assertTrue(len(result[0]) == 0)
        self.assertTrue(len(result[1]) == 0)

    def test_direct_connections_non_empty_sets(self):
        l = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        df = tu.sample_group_df(l)
        df['user_id'] = df['com_id']
        df['com_id'] = range(len(df))
        df['text_id'] = range(100, 100 + len(df))
        rels = [('posts', 'user', 'user_id'), ('intext', 'text', 'text_id')]

        result = self.test_obj.direct_connections(df, 2, rels)

        self.assertTrue(result[0] == set({2, 3}))
        self.assertTrue(result[1] == set({'posts'}))

    def test_all_connections_no_connections(self):
        df = tu.sample_df(10)
        df.columns = ['com_id', 'user_id']
        rels = [('posts', 'user', 'user_id')]

        result = self.test_obj.all_connections(df, 2, rels)

        self.assertTrue(result == (set({2}), set()))

    def test_all_connections_with_direct_connections(self):
        l = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        df = tu.sample_group_df(l)
        df['user_id'] = df['com_id']
        df['com_id'] = range(len(df))
        df['text_id'] = range(100, 100 + len(df))
        rels = [('posts', 'user', 'user_id'), ('intext', 'text', 'text_id')]

        result = self.test_obj.all_connections(df, 2, rels)

        self.assertTrue(result == (set({2, 3}), set({'posts'})))

    def test_all_connections_with_indirect_connections(self):
        l = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        df = tu.sample_group_df(l)
        df['user_id'] = df['com_id']
        df['com_id'] = range(len(df))
        df['text_id'] = [100, 101, 102, 103, 104, 105, 106, 107, 103, 109]
        rels = [('posts', 'user', 'user_id'), ('intext', 'text', 'text_id')]

        result = self.test_obj.all_connections(df, 2, rels)

        self.assertTrue(result[0] == set({8, 9, 2, 3}))
        self.assertTrue(result[1] == set({'intext', 'posts'}))


def test_suite():
    s = unittest.TestLoader().loadTestsFromTestCase(ConnectionsTestCase)
    return s

if __name__ == '__main__':
    unittest.main()
