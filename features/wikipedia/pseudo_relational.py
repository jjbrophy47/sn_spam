"""
This module contains an API to generate psuedo-relational features for wikipedia.
"""
import numpy as np
import pandas as pd


def pseudo_relational_features(y_hat, user_ids, relations):
    """Generates pseudo-relational features using given predictions and relations."""

    relation_dir = 'data/wikipedia/processed/relation_files/'

    Xr = []
    Xr_cols = []

    for relation_id in relations:

        if relation_id == 'link_id':
            csv_name = 'link_relation.csv'
        elif relation_id == 'ores_id':
            csv_name = 'ores_relation.csv'
        elif relation_id == 'edit_id':
            csv_name = 'edit_relation.csv'
        elif relation_id == 'burst_id':
            csv_name = 'burst_relation.csv'

        fname = '%s%s' % (relation_dir, csv_name)
        x, name = _relation(y_hat, user_ids, fname, relation_id)
        Xr.append(x)
        Xr_cols.append(name)

    Xr = np.hstack(Xr)

    return Xr, Xr_cols


def _relation(y_hat, user_ids, fname, relation_id):

    feature_name = relation_id.split('_')[0] + '_relation'

    # get relational info
    rel_df = pd.read_csv(fname)
    rel_df = rel_df[['user_id', relation_id]]

    # merge predictions with relational information
    data_df = pd.DataFrame(list(zip(y_hat, user_ids)), columns=['y_hat', 'user_id'])

    # compute mean prediction for each relational hub
    hub_means = []
    for hub_id, hub_df in rel_df.groupby(relation_id):
        users = hub_df['user_id']
        hub_mean = data_df[data_df['user_id'].isin(users)]['y_hat'].mean()
        hub_means.append((hub_id, hub_mean))
    hub_means_df = pd.DataFrame(hub_means, columns=[relation_id, 'mean_y_hat'])

    new_rel_df = rel_df.merge(hub_means_df, on=relation_id)

    # compute mean prediction for each user
    feature_vals = []
    for user_id in data_df['user_id']:
        feature_vals.append(new_rel_df[new_rel_df['user_id'] == user_id]['mean_y_hat'].mean())

    data_df[feature_name] = feature_vals
    data_df[feature_name] = data_df[feature_name].fillna(data_df['y_hat'])
    assert len(feature_vals) == len(data_df)

    return data_df[feature_name].to_numpy().reshape(-1, 1), feature_name
