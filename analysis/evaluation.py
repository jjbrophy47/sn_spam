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

        preds = self.read_predictions(test_df, ind_pred_f, rel_pred_f)
        modified_df = self.read_modified(data_f) if modified else None

        score_dict = {}
        fname = image_f + 'pr_' + fold
        for pred in preds:
            pred_df, col, name, line = pred
            save = True if pred[1] in preds[-1][1] else False
            scores = self.merge_and_score(test_df, pred, fname, save,
                                          modified_df)
            score_dict[name] = scores
        return score_dict

    # private
    def settings(self):
        noise_limit = 0.000025
        self.util_obj.set_noise_limit(noise_limit)

    def file_folders(self):
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

    def read_predictions(self, test_df, ind_pred_f, rel_pred_f, dset='test'):
        fold = self.config_obj.fold
        util = self.util_obj
        fname = dset + '_' + fold
        preds = []

        nps_df = util.read_csv(ind_pred_f + 'nps_' + fname + '_preds.csv')
        ind_df = util.read_csv(ind_pred_f + fname + '_preds.csv')
        psl_df = util.read_csv(rel_pred_f + 'psl_preds_' + fold + '.csv')
        mrf_df = util.read_csv(rel_pred_f + 'mrf_preds_' + fold + '.csv')

        if nps_df is not None and len(nps_df) == len(test_df):
            preds.append((nps_df, 'nps_pred', 'no_pseudo', '-'))
        if ind_df is not None and len(ind_df) == len(test_df):
            preds.append((ind_df, 'ind_pred', 'ind', '--'))
        if psl_df is not None and len(psl_df) == len(test_df):
            preds.append((psl_df, 'psl_pred', 'psl', ':'))
        if mrf_df is not None and len(mrf_df) == len(test_df):
            preds.append((mrf_df, 'mrf_pred', 'mrf', '-.'))

        return preds

    def merge_and_score(self, test_df, pred, fname, save=False,
                        modified_df=None):
        pred_df, col, name, line = pred

        merged_df = self.merge_predictions(test_df, pred_df)

        if modified_df is not None:
            merged_df = self.filter(merged_df, modified_df)

        noise_df = self.apply_noise(merged_df, col)
        pr, roc, r, p, npr = self.compute_scores(noise_df, col)
        self.print_scores(name, pr, roc, npr)
        # self.util_obj.plot_pr_curve(name, fname, r, p, npr, line=line,
        #         save=save)
        scores = {'aupr': round(pr, 7), 'auroc': round(roc, 7),
                  'naupr': round(npr, 7)}
        return scores

    def merge_predictions(self, test_df, pred_df):
        merged_df = test_df.merge(pred_df, on='com_id', how='left')
        return merged_df

    def read_modified(self, data_f):
        fname = data_f + 'labels.csv'
        if self.util_obj.check_file(fname):
            modified_df = pd.read_csv(fname)
        return modified_df

    def filter(self, merged_df, modified_df):
        temp_df = merged_df[~merged_df['com_id'].isin(modified_df['com_id'])]
        return temp_df

    def apply_noise(self, merged_df, col):
        merged_df[col] = merged_df[col].apply(self.util_obj.gen_noise)
        return merged_df

    def compute_scores(self, pf, col):
        fpr, tpr, _ = roc_curve(pf['label'], pf[col])
        prec, rec, _ = precision_recall_curve(pf['label'], pf[col])
        nPre, nRec, _ = precision_recall_curve(pf['label'], 1 - pf[col],
                                               pos_label=0)
        auroc, aupr, nAupr = auc(fpr, tpr), auc(rec, prec), auc(nRec, nPre)
        return aupr, auroc, rec, prec, nAupr

    def print_scores(self, name, aupr, auroc, naupr, fw=None):
        s = name + ' evaluation...AUPR: %.4f, AUROC: %.4f, N-AUPR: %.4f' + '\n'
        self.util_obj.write(s % (aupr, auroc, naupr), fw=fw)
