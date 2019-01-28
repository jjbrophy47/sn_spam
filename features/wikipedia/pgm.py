"""
This module contains an API to create the appropriate libra files for MRF LBP for Wikipedia.
"""
import pandas as pd


def create_files(y_hat, user_ids, relations):
    """Generates pseudo-relational features using given predictions and relations."""

    relation_dir = 'data/wikipedia/processed/relation_files/'

    # classification nodes
    target_name = 'user_id'
    target_priors = list(zip(user_ids, y_hat))

    # relational nodes
    relations_dict = {}

    for relation_id in relations:

        if relation_id == 'link_id':
            csv_name = 'link_relation.csv'
        elif relation_id == 'ores_id':
            csv_name = 'ores_relation.csv'
        elif relation_id == 'edit_id':
            csv_name = 'edit_relation.csv'

        fname = '%s%s' % (relation_dir, csv_name)
        df = pd.read_csv(fname)
        connections = list(zip(df[relation_id], df[target_name]))
        relations_dict[relation_id] = connections

    return target_priors, relations_dict, target_name
