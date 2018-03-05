"""
This module evaluates the predictions from the independent and relational
models.
"""
import os
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc


class Evaluation:

    def __init__(self, config_obj, generator_obj, util_obj):
        self.config_obj = config_obj
        self.generator_obj = generator_obj
        self.util_obj = util_obj

    # public
    def evaluate(self, test_df, modified=False):
        """Evaluation of both the indeendent and relational models.
        test_df: testing dataframe."""
        fold = self.config_obj.fold
        test_df = test_df.copy()

        self.settings()
        data_f, ind_pred_f, rel_pred_f, image_f, status_f = self.file_folders()
        sw = self.open_status_writer(status_f)

        preds = self.read_predictions(test_df, ind_pred_f, rel_pred_f)
        modified_df = self.read_modified(data_f) if modified else None

        score_dict = {}
        fname = image_f + 'pr_' + fold
        for pred in preds:
            pred_df, col, name, line = pred
            save = True if pred[1] in preds[-1][1] else False
            scores = self.merge_and_score(test_df, pred, fname, save,
                                          modified_df, sw)
            score_dict[name] = scores
        self.util_obj.close_writer(sw)
        return score_dict

    # private
    def settings(self):
        """Sets the noise limit added to predictions."""
        noise_limit = 0.0025
        self.util_obj.set_noise_limit(noise_limit)

    def file_folders(self):
        """Returns absolute path directories."""
        ind_dir = self.config_obj.ind_dir
        rel_dir = self.config_obj.rel_dir
        domain = self.config_obj.domain

        ind_data_f = ind_dir + 'data/' + domain + '/'
        ind_pred_f = ind_dir + 'output/' + domain + '/predictions/'
        rel_out_f = rel_dir + 'output/' + domain + '/'
        rel_pred_f = rel_out_f + 'predictions/'
        rel_image_f = rel_out_f + 'images/'
        status_f = rel_dir + 'output/' + domain + '/status/'
        if not os.path.exists(rel_image_f):
            os.makedirs(rel_image_f)
        if not os.path.exists(status_f):
            os.makedirs(status_f)
        return ind_data_f, ind_pred_f, rel_pred_f, rel_image_f, status_f

    def open_status_writer(self, status_f):
        """Opens a file object to write evaluation status to.
        status_f: status folder.
        Returns file object."""
        fold = self.config_obj.fold
        fname = status_f + 'eval_' + fold + '.txt'
        f = self.util_obj.open_writer(fname)
        return f

    def read_predictions(self, test_df, ind_pred_f, rel_pred_f, dset='test'):
        """Reads in the independent and relational model predictions.
        ind_pred_f: independent model predictions folder.
        rel_pred_f: relational model predictions folder.
        dset: dataset to read.
        Returns dataframes for the predictions."""
        fold = self.config_obj.fold
        util = self.util_obj
        fname = dset + '_' + fold
        preds = []

        nps_df = util.read_csv(ind_pred_f + 'nps_' + fname + '_preds.csv')
        ind_df = util.read_csv(ind_pred_f + fname + '_preds.csv')
        psl_df = util.read_csv(rel_pred_f + 'psl_preds_' + fold + '.csv')
        mrf_df = util.read_csv(rel_pred_f + 'mrf_preds_' + fold + '.csv')

        if nps_df is not None and len(nps_df) == len(test_df):
            preds.append((nps_df, 'nps_pred', 'No Pseudo', '-'))
        if ind_df is not None and len(ind_df) == len(test_df):
            preds.append((ind_df, 'ind_pred', 'Independent', '--'))
        if psl_df is not None and len(psl_df) == len(test_df):
            preds.append((psl_df, 'psl_pred', 'PSL', ':'))
        if mrf_df is not None and len(mrf_df) == len(test_df):
            preds.append((mrf_df, 'mrf_pred', 'MRF', '-.'))

        return preds

    def merge_and_score(self, test_df, pred, fname, save=False,
                        modified_df=None, fw=None):
        """Merges the predictions onto the test set and computes the
                evaluation metrics.
        test_df: test set dataframe.
        pred: tuple with prediction dataframe, col, name, and line pattern.
        fname: name of the file to save the pr curve.
        save: boolean to save pr curve or not.
        modified_df: if true, take out relabeled instances."""
        pred_df, col, name, line = pred

        merged_df = self.merge_predictions(test_df, pred_df)

        if modified_df is not None:
            merged_df = self.filter(merged_df, modified_df)

        noise_df = self.apply_noise(merged_df, col)
        pr, roc, r, p, npr = self.compute_scores(noise_df, col)
        self.print_scores(name, pr, roc, npr, fw=fw)
        # self.util_obj.plot_pr_curve(name, fname, r, p, npr, line=line,
        #         save=save)
        scores = (pr, roc, npr)
        return scores

    def merge_predictions(self, test_df, pred_df):
        """Merges the independent and relational dataframes together.
        test_df: testing dataframe.
        pred_df: predictions dataframe.
        Returns merged dataframe."""
        merged_df = test_df.merge(pred_df, on='com_id', how='left')
        return merged_df

    def read_modified(self, data_f):
        """Read in the predictions for the adversarial comments.
        pred_f: predictions folder.
        Returns dataframe of identified adversarial comments, dataframe of
                false positives."""
        fname = data_f + 'labels.csv'
        if self.util_obj.check_file(fname):
            modified_df = pd.read_csv(fname)
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

    def apply_noise(self, merged_df, col):
        """Adds a small amount of noise to each prediction for both models.
        merged_df: testing dataframe with merged predictions."""
        merged_df[col] = merged_df[col].apply(self.util_obj.gen_noise)
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

    def print_scores(self, name, aupr, auroc, naupr, fw=None):
        """Print the scores to stdout.
        name: name of the model.
        aupr: area under the pr curve.
        auroc: area under the roc curve.
        naupr: neg-aupr."""
        s = name + ' evaluation...AUPR: %.4f, AUROC: %.4f, N-AUPR: %.4f' + '\n'
        self.util_obj.write(s % (aupr, auroc, naupr), fw=fw)
