"""
This module evaluates the predictions from the independent and relational
models.
"""
import os
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc


class Evaluation:
    """This class handles all operations pertaining to the evaluation
    performance of the independent and relational models."""

    def __init__(self, config_obj, generator_obj, util_obj):
        """Initialize object dependencies."""

        self.config_obj = config_obj
        """User settings."""
        self.generator_obj = generator_obj
        """Finds and generates relational ids."""
        self.util_obj = util_obj
        """Utility methods."""

    # public
    def evaluate(self, test_df):
        """Evaluation of both the indeendent and relational models.
        test_df: testing dataframe."""
        fold = self.config_obj.fold
        test_df = test_df.copy()

        self.settings()
        data_f, ind_pred_f, rel_pred_f, image_f = self.define_file_folders()
        ind_df, rel_df = self.read_predictions(ind_pred_f, rel_pred_f)
        merged_df = self.merge_predictions(test_df, ind_df, rel_df)
        noise_df = self.apply_noise(merged_df)

        ipr, iroc, ir, ip, inpr = self.compute_scores(noise_df, 'ind_pred')
        rpr, rroc, rr, rp, rnpr = self.compute_scores(noise_df, 'rel_pred')

        fname = image_f + 'pr_' + fold
        self.print_scores('Independent', ipr, iroc, inpr)
        self.print_scores('Relational', rpr, rroc, rnpr)
        self.util_obj.plot_pr_curve('Independent', '', ir, ip, ipr)
        self.util_obj.plot_pr_curve('Relational', fname, rr, rp, rpr,
                line='--', save=True)

    def evaluate_modified(self, test_df):
        """Evaluation of the entire test set, with only relational predictions
                that helped or hurt, all others are independent predictions.
        test_df: testing dataframe."""
        fold = self.config_obj.fold
        test_df = test_df.copy()

        self.settings()
        data_f, ind_pred_f, rel_pred_f, image_f = self.define_file_folders()
        ind_df, rel_df = self.read_predictions(ind_pred_f, rel_pred_f)
        merged_df = self.merge_predictions(test_df, ind_df, rel_df)
        modified_df = self.read_modified(data_f)
        filt_df = self.filter(merged_df, modified_df)
        noise_df = self.apply_noise(filt_df)

        ipr, iroc, ir, ip, inpr = self.compute_scores(noise_df, 'ind_pred')
        rpr, rroc, rr, rp, rnpr = self.compute_scores(noise_df, 'rel_pred')

        fname = image_f + 'pr_' + fold
        self.print_scores('Independent', ipr, iroc, inpr)
        self.print_scores('Relational', rpr, rroc, rnpr)
        self.util_obj.plot_pr_curve('Independent', '', ir, ip, ipr)
        self.util_obj.plot_pr_curve('Relational', fname, rr, rp, rpr,
                line='--', save=True)

    # private
    def settings(self):
        """Sets the noise limit added to predictions."""
        noise_limit = 0.0025
        self.util_obj.set_noise_limit(noise_limit)

    def define_file_folders(self):
        """Returns absolute path directories."""
        ind_dir = self.config_obj.ind_dir
        rel_dir = self.config_obj.rel_dir
        domain = self.config_obj.domain

        ind_data_f = ind_dir + 'data/' + domain + '/'
        ind_pred_f = ind_dir + 'output/' + domain + '/predictions/'
        rel_out_f = rel_dir + 'output/' + domain + '/'
        rel_pred_f = rel_out_f + 'predictions/'
        rel_image_f = rel_out_f + 'images/'
        if not os.path.exists(rel_image_f):
            os.makedirs(rel_image_f)
        return ind_data_f, ind_pred_f, rel_pred_f, rel_image_f

    def read_predictions(self, ind_pred_f, rel_pred_f):
        """Reads in the independent and relational model predictions.
        ind_pred_f: independent model predictions folder.
        rel_pred_f: relational model predictions folder.
        Returns dataframes for the predictions."""
        fold = self.config_obj.fold
        dset = 'test_'

        ind_df = pd.read_csv(ind_pred_f + dset + fold + '_preds.csv')
        rel_df = pd.read_csv(rel_pred_f + 'predictions_' + fold + '.csv')
        return ind_df, rel_df

    def merge_predictions(self, test_df, ind_df, rel_df):
        """Merges the independent and relational dataframes together.
        test_df: testing dataframe.
        ind_df: independent predictions dataframe.
        rel_df: relational predictions dataframe.
        Returns merged dataframe."""
        merged_df = test_df.merge(ind_df)
        merged_df = merged_df.merge(rel_df)
        return merged_df

    def read_modified(self, data_f):
        """Read in the predictions for the adversarial comments.
        pred_f: predictions folder.
        Returns dataframe of identified adversarial comments, dataframe of
                false positives."""
        modified_df = pd.read_csv(data_f + 'labels.csv')
        return modified_df

    def filter(self, merged_df, modified_df):
        """Only keep the relational predictions for those in helped_df or
                hurt_df dataframes.
        df: comments dataframe with independent predictions.
        helped_df: helped dataframe with relational predictions.
        hurt_df: hurt dataframe with relational predictions.
        Returns dataframe with an altered relational predictions column."""
        temp_df = merged_df[~merged_df['com_id'].isin(modified_df['com_id'])]
        return temp_df

    def apply_noise(self, merged_df):
        """Adds a small amount of noise to each prediction for both models.
        merged_df: testing dataframe with merged predictions."""
        gen_noise = self.util_obj.gen_noise
        merged_df['ind_pred'] = merged_df['ind_pred'].apply(gen_noise)
        merged_df['rel_pred'] = merged_df['rel_pred'].apply(gen_noise)
        return merged_df

    def compute_scores(self, pf, col):
        """Computes the precision, recall, aupr, and auroc scores.
        pf: dataframe with the merged predictions.
        col: column identifer for predictions to compare to the labels.
        Returns aupr, auroc, recalls, precisions, and neg-aupr scores."""
        fpr, tpr, _ = roc_curve(pf['label'], pf[col])
        prec, rec, _ = precision_recall_curve(pf['label'], pf[col])
        nPre, nRec, _ = precision_recall_curve(pf['label'], 1 - pf[col],
                pos_label=0)
        auroc, aupr, nAupr = auc(fpr, tpr), auc(rec, prec), auc(nRec, nPre)
        return aupr, auroc, rec, prec, nAupr

    def print_scores(self, name, aupr, auroc, naupr):
        """Print the scores to stdout.
        name: name of the model.
        aupr: area under the pr curve.
        auroc: area under the roc curve.
        naupr: neg-aupr."""
        s = '\t' + name + ' evaluation...AUPR: %.4f, AUROC: %.4f, N-AUPR: %.4f'
        print(s % (aupr, auroc, naupr))

    def gen_group_ids(self, df):
        """Generates any missing group_id columns.
        df: comments dataframe with predictions.
        Returns dataframe with filled in group_ids."""
        for relation, group, group_id in self.config_obj.relations:
            df = self.generator_obj.gen_group_id(df, group_id)
        return df

    def relational_comments_only(self, df):
        """Excludes comments that do not have any relations in the dataset.
        df: comments dataframe.
        Returns dataframe with comments that contain a relation."""
        possible_relations = self.config_obj.relations
        include_ids = set()

        for relation, group, group_id in possible_relations:
            g_df = df.groupby(group_id).size().reset_index()
            g_df.columns = [group_id, 'size']
            g_df = g_df[g_df['size'] > 1]
            r_df = df.merge(g_df, on=group_id)
            include_ids.update(r_df['com_id'])

        filtered_df = df[df['com_id'].isin(include_ids)]
        return filtered_df
