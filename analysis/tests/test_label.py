"""
Tests the pred_builder module.
"""
import unittest
import pandas as pd
import mock
from .context import label
from .context import config
from .context import generator
from .context import test_utils as tu


class LabelTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        mock_generator_obj = mock.Mock(generator.Generator)
        self.test_obj = label.Label(config_obj, mock_generator_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        # setup
        result = self.test_obj

        # assert
        self.assertTrue(isinstance(result.config_obj, config.Config))
        self.assertTrue(isinstance(result.generator_obj, generator.Generator))

    def test_relabel(self):
        self.test_obj.config_obj.relations = [('intext', 'text', 'text_id'),
                ('invideo', 'video', 'vid_id'),
                ('posts', 'user', 'user_id'),
                ('inment', 'mention', 'ment_id')]
        all_relations = self.test_obj.config_obj.relations
        relations = [('intext', 'text', 'text_id'),
                ('posts', 'user', 'user_id')]
        l_dict = {'moo': 'cow'}
        self.test_obj.define_file_folders = mock.Mock(return_value='data/')
        self.test_obj.read_comments = mock.Mock(return_value='df2')
        self.test_obj.filter_relations = mock.Mock(return_value=relations)
        self.test_obj.generator_obj.gen_group_ids = mock.Mock(
                return_value='f_df')
        self.test_obj.relabel_relations = mock.Mock(return_value=l_dict)
        self.test_obj.merge_labels = mock.Mock(return_value=('new_df', 'l_df'))
        self.test_obj.write_new_dataframe = mock.Mock()

        self.test_obj.relabel('df')

        self.test_obj.define_file_folders.assert_called()
        self.test_obj.read_comments.assert_called_with('df', 'data/')
        self.test_obj.filter_relations.assert_called_with(all_relations)
        self.test_obj.generator_obj.gen_group_ids.assert_called_with('df2',
                relations)
        self.test_obj.relabel_relations.assert_called_with('f_df', relations)
        self.test_obj.merge_labels.assert_called_with('f_df', l_dict)
        self.test_obj.write_new_dataframe.assert_called_with('new_df', 'l_df',
                'data/')

    def test_define_file_folders(self):
        result = self.test_obj.define_file_folders()

        self.assertTrue(result == 'ind/data/soundcloud/')

    def test_read_comments(self):
        df = tu.sample_df(10)
        pd.read_csv = mock.Mock(return_value=df)
        self.test_obj.config_obj.start = 2
        self.test_obj.config_obj.end = 1000

        result = self.test_obj.read_comments(None, 'data/')

        pd.read_csv.assert_called_with('data/comments.csv', nrows=1000)
        self.assertTrue(len(result) == 8)

    def test_filter_relations(self):
        relations = [('intext', 'text', 'text_id'),
                ('invideo', 'video', 'vid_id'),
                ('posts', 'user', 'user_id'),
                ('inment', 'mention', 'ment_id')]

        result = self.test_obj.filter_relations(relations)

        exp = [('intext', 'text', 'text_id'), ('posts', 'user', 'user_id')]
        self.assertTrue(len(result) == 2)
        self.assertTrue(result == exp)

    def test_relabel_relations(self):
        g_df = tu.sample_group_df()
        g_df.columns = ['text_id']
        relations = [self.test_obj.config_obj.relations[0]]
        rel_dict = {'booger': 'sneeze'}
        self.test_obj.relabel_groups = mock.Mock(return_value=rel_dict)

        result = self.test_obj.relabel_relations(g_df, relations)

        self.test_obj.relabel_groups.assert_called_with(g_df, 'text_id',
                [1, 2, 3])
        self.assertTrue(result == rel_dict)

    def test_relabel_groups(self):
        g_df = tu.sample_group_df()
        g_df.columns = ['text_id']
        self.test_obj.relabel_group = mock.Mock()
        self.test_obj.relabel_group.side_effect = [{'a': 1}, {'b': 69}]

        result = self.test_obj.relabel_groups(g_df, 'text_id', [1, 2])

        exp = {'a': 1, 'b': 69}
        self.assertTrue(self.test_obj.relabel_group.call_count == 2)
        self.assertTrue(result == exp)

    def test_relabel_group(self):
        g_df = tu.sample_df(8)
        g_df['label'] = [1, 1, 0, 1, 0, 1, 1, 0]

        result = self.test_obj.relabel_group(g_df)

        self.assertTrue(result == {2: 1, 4: 1, 7: 1})

    def test_merge_labels(self):
        df = tu.sample_df(2)
        labels1_df = tu.sample_df(2)
        labels2_df = tu.sample_df(2)
        new_df = tu.sample_df(2)
        temp_df = tu.sample_df(2)
        temp_df.columns = ['com_id', 'label']
        temp_df['new_label'] = pd.Series([1, 1])
        df.copy = mock.Mock(return_value=new_df)
        df.merge = mock.Mock(return_value=temp_df)
        labels1_df.reset_index = mock.Mock(return_value=labels2_df)
        pd.DataFrame.from_dict = mock.Mock(return_value=labels1_df)

        result = self.test_obj.merge_labels(df, {})

        labels2_df.columns = ['com_id', 'new_label']
        temp_df['new_label'] = pd.Series([1, 1])
        new_df['label'] = temp_df['new_label'].apply(int)
        df.copy.assert_called()
        pd.DataFrame.from_dict.assert_called_with({}, orient='index')
        labels1_df.reset_index.assert_called()
        df.merge.assert_called_with(labels2_df, on='com_id', how='left')
        self.assertTrue(result[0].equals(new_df))
        self.assertTrue(result[1].equals(labels2_df))

    def test_write_new_dataframe(self):
        new_df = tu.sample_df(10)
        labels_df = tu.sample_df(10)
        new_df.to_csv = mock.Mock()
        labels_df.to_csv = mock.Mock()

        self.test_obj.write_new_dataframe(new_df, labels_df, 'data/')

        new_df.to_csv.assert_called_with('data/modified.csv',
            encoding='utf-8', line_terminator='\n', index=None)
        labels_df.to_csv.assert_called_with('data/labels.csv', index=None)


def test_suite():
    s = unittest.TestLoader().loadTestsFromTestCase(LabelTestCase)
    return s

if __name__ == '__main__':
    unittest.main()
