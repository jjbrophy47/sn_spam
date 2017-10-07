"""
Module of utility methods.
"""
import os
import sys
import time
import pickle
import random
import scipy.sparse
import numpy as np
import pandas as pd
import termcolor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class Util:
    """Class that handles a range of tasks from plotting, to generating noise,
    to printing and plotting. Methods are alphabetized."""

    def __init__(self):
        """Initialize class attributes."""

        self.noise_limit = 0.0025
        """Limit on the amount of noise that can be added."""
        self.timer = []
        """Stack of start times to keep track of."""

    # public
    def check_file(self, file):
        """Checks to see if the file exists
        file: path of the file.
        Returns True if it exists, exits the application if not."""
        if os.path.exists(file):
            return True
        else:
            self.exit('cannot read ' + file)

    def classify(self, data, fold, feat_set, image_f, pred_f, model_f,
            save_pr_plot=True, line='-', save_feat_plot=True, save_preds=True,
            classifier='rf', dset='test', fw=None):
        """Method to independently classify instances.
        data: tuple containing training and testing data, test set ids,
                and a list of feature names.
        fold: experiment identifier.
        feat_set: name pertaining to all features.
        image_f: image folder.
        pred_f: predictions folder.
        model_f: model folder.
        save_pr_plot: boolean to save aupr plot.
        line: line pattern to use on aupr plot.
        save_feat_plot: boolean to save feature plot.
        save_preds: boolean to save predictions or not.
        classifier: name of classifier, options are: 'lr' and 'rf'.
        dset: dataset to classify (e.g. 'val' or 'test').
        fw: file writer object."""
        model_name = feat_set + '_' + fold
        model_file = dset + '_' + classifier + '_' + fold + '.pkl'
        x_tr, y_tr, x_te, y_te, id_te, feat_names = data

        self.start('training...', fw=fw)
        model = self.classifier(classifier)
        model = model.fit(x_tr, y_tr)
        self.save(model, model_f + model_file)
        self.end(fw=fw)

        self.start('testing...', fw=fw)
        test_probs = model.predict_proba(x_te)
        self.end(fw=fw)

        self.start('evaluating...', fw=fw)
        auroc, aupr, p, r, mp, mr, t = self.compute_scores(test_probs, y_te)
        self.end(fw=fw)

        self.print_scores(mp, mr, t, aupr, auroc, fw=fw)
        self.print_median_mean(id_te, test_probs, y_te, fw=fw)

        fname = image_f + model_name
        if dset == 'test':
            self.plot_pr_curve(model_name, fname, r, p, aupr, title=feat_set,
                    line=line, save=save_pr_plot)
            if save_feat_plot:
                self.plot_features(model, classifier, feat_names, fname,
                        save=save_feat_plot)
        if save_preds:
            self.save_preds(test_probs, id_te, fold, pred_f, dset)

    def close_writer(self, sw):
        """Closes a file writer.
        sw: file writer object."""
        sw.close()

    def colorize(self, string, color, display):
        """Gives the string the specified color if there is a display.
        string: string to colorize.
        color: color to give the string.
        display: boolean indicating if the application is run on a consolde.
        Returns a colorized string if there is a display, string otherwise."""
        s = string

        if display:
            s = termcolor.colored(string, color)
        return s

    def div0(self, num, denom):
        """Divide operation that deals with a 0 value denominator.
        num: numerator.
        denom: denominator.
        Returns 0.0 if the denominator is 0, otherwise returns a float."""
        return 0.0 if denom == 0 else float(num) / denom

    def end(self, message='', fw=None):
        """Pop a start time and take the time difference from now.
        message: message to print."""
        unit = 's'
        elapsed = time.time() - self.timer.pop()
        if elapsed >= 60:
            elapsed /= 60
            unit = 'm'
        s = message + '%.2f' + unit + '\n'
        if fw is not None:
            fw.write(s % (elapsed))
        else:
            self.out(s % (elapsed))

    def exit(self, message='Unexpected error occurred!'):
        """Convenience method to fail gracefully.
        message: messaage to display to the user as to the error."""
        print(message)
        print('exiting...')
        exit(0)

    def file_len(self, fname):
        """Counts the number of lines in a file.
        fname: path of the file.
        Returns the number of lines in the specified file."""
        lines = 0

        f = open(fname, 'r')
        lines = len(f.readlines())
        f.close()
        return lines

    def gen_noise(self, pred):
        """Returns a prediction with some noise added to it.
        pred: predicion (e.g. value between 0.0 and 1.0).
        Returns predictions with noise."""
        noise = random.uniform(-self.noise_limit, self.noise_limit)
        result = max(0.0, min(1.0, pred + noise))
        return result

    def get_comments_filename(self, modified):
        """Chooses the correct comments file to read
        modified: Boolean indicating to read the modified comments file.
        Returns the name of the appropriate comments file."""
        filename = 'comments.csv'
        if modified:
            filename = 'modified.csv'
        return filename

    def load(self, filename):
        """Loads a binary pickled object.
        filename: path of the file.
        Returns loaded object."""
        if self.check_file(filename):
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
        return obj

    def load_sparse(self, filename):
        """Loads a sparse matrix object.
        filename: path to the sparse matrix object file.
        Returns sparse matrix object."""
        matrix = scipy.sparse.load_npz(filename)
        return matrix

    def mean(self, numbers):
        """Computes the mean for a list of numbers.
        numbers: list of numbers.
        Returns mean as a float."""
        return np.mean(numbers)

    def out(self, message):
        """Custom print method to print multiple times on one line.
        message: string to print immediately."""
        sys.stdout.write(message)
        sys.stdout.flush()

    def open_writer(self, name, mode='w'):
        f = open(name, mode)
        return f

    def percent(self, num, denom):
        """Turns fraction into a percent.
        num: numerator.
        denom: denominator.
        Returns float in percent form."""
        return self.div0(num, denom) * 100.0

    def plot_features(self, model, classifier, features, fname, save=False):
        """Plots relative feature importance.
        model: fitted model.
        classifier: specific model (e.g. 'lr', 'rf').
        features: list of feature names.
        fname: filename of where to store the plot.
        save: boolean of whether the plot should be saved."""
        if classifier == 'lr':
            feat_importance = model.coef_[0]
        elif classifier == 'rf':
            feat_importance = model.feature_importances_

        # normalize and rearrange features
        feat_norm = 100.0 * (feat_importance / feat_importance.max())
        sorted_idx = np.argsort(feat_norm)
        pos = np.arange(sorted_idx.shape[0]) + 0.5  # [0.5, 1.5, ...]
        feat_importance_sort = feat_importance[sorted_idx]
        feat_sort = np.asanyarray(features)[sorted_idx]

        # plot relative feature importance
        color = '#7A68A6'
        plt.figure(figsize=(12, 12))
        plt.barh(pos, feat_importance_sort, align='center', color=color)
        plt.yticks(pos, feat_sort)
        plt.xlabel('Relative Importance')
        plt.title('Feature Importance')
        if save:
            plt.savefig(fname + '_feats.png', bbox_inches='tight')

    def plot_pr_curve(self, model, fname, rec, prec, aupr, title='',
            line='-', save=False, show_legend=False, show_grid=False,
            more_ticks=False):
        """Plots a precision-recall curve.
        model: name of the model.
        fname: filename to save the plot.
        rec: recalls from the aupr.
        prec: precisions from the aupr.
        aupr: area under the pr curve.
        title: title of the plot.
        line: shape used to draw the curve.
        save: boolean specifying whether to save the plot."""
        self.set_plot_rc()
        plt.figure(2)
        plt.plot(rec, prec, line, label=model + ' = %0.3f' % aupr)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title(title, fontsize=22)
        plt.xlabel('Recall', fontsize=22)
        plt.ylabel('Precision', fontsize=22)
        plt.tick_params(axis='both', labelsize=18)

        if show_legend:
            plt.legend(loc='lower left', prop={'size': 6})

        if show_grid:
            ax = plt.gca()
            ax.grid(b=True, which='major', color='#E5DCDA', linestyle='-')

        if more_ticks:
            plt.yticks(np.arange(0.0, 1.01, 0.1))
            plt.xticks(np.arange(0.0, 1.01, 0.1), rotation=70)

        if save:
            plt.savefig(fname + '.png', bbox_inches='tight')
            plt.clf()

    def print_stats(self, df, r_df, relation, dset, fw=None):
        """Prints information about a relationship in the data.
        df: comments dataframe.
        r_df: df containing number of times relationship occurred.
        relation: name of relation (e.g. posts).
        dset: dataset (e.g. 'val' or 'test')."""
        spam = r_df['label'].sum()
        out_str = '\n\t[' + dset + '] ' + relation + ': >1: ' + str(len(r_df))
        out_str += ', spam: ' + str(spam)
        self.write(out_str, fw=fw)

    def read_csv(self, filename):
        """Safe read for pandas dataframes.
        filename: path to data file.
        Returns dataframe if the file exists, None otherwise."""
        result = None

        if os.path.exists(filename):
            result = pd.read_csv(filename)
        return result

    def save(self, obj, filename):
        """Pickles an object to a binary file.
        obj: object to pickle.
        filename: path of the file."""
        with open(filename, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def save_sparse(self, matrix, filename):
        """Saves a sparse matrix object to a file.
        matrix: sparse matrix object.
        filename: path to the file to save the object to."""
        scipy.sparse.save_npz(filename, matrix)

    def set_noise_limit(self, noise_limit):
        """Setter for noise_limit."""
        self.noise_limit = noise_limit

    def start(self, message='', fw=None):
        """Pushes a start time onto a stack and print a specified message.
        message: message to print."""
        self.write(message=message, fw=fw)
        self.timer.append(time.time())

    def write(self, message='', fw=None):
        if fw is not None:
            fw.write(message)
        else:
            self.out(message)

    # private
    def classifier(self, classifier):
        """Instantiates the desired classifier.
        classifier: model to classify with (e.g. 'rf', 'lr').
        Returns instantiated sklearn classifier."""
        if classifier == 'rf':
            model = RandomForestClassifier(n_estimators=100, max_depth=4)
        elif classifier == 'lr':
            model = LogisticRegression()
        return model

    def compute_scores(self, probs, y):
        """Generates noisy predictions and computes various metrics.
        probs: predictions, shape=(2, <num_instances>).
        y: list of true labels.
        report: file to write performance to.
        dset: dataset (e.g. 'train', 'val', 'test').
        Returns auroc, aupr, recalls, precisions, max precision, max recall,
                and threshold where those max values take place."""
        prob_preds_noise = [self.gen_noise(pred) for pred in probs[:, 1]]
        fpr, tpr, tholds = sm.roc_curve(y, prob_preds_noise)
        prec, rec, tholds = sm.precision_recall_curve(y, prob_preds_noise)
        aupr = sm.average_precision_score(y, prob_preds_noise)
        auroc = sm.auc(fpr, tpr)
        max_p, max_r, thold = self.find_max_prec_recall(prec, rec, tholds)
        return auroc, aupr, prec, rec, max_p, max_r, thold

    def find_max_prec_recall(self, prec, rec, tholds):
        """Finds the precision and recall scores with the maximum amount of
        area and returns their values, including the threshold.
        prec: list of precisions from the pr curve.
        rec: list of recalls from the pr curve.
        tholds: list of thresholds from the pr curve.
        Returns max precision and recall scores, including their threshold."""
        max_val, max_prec, max_rec, max_thold = -1, -1, -1, None

        for i in range(len(prec)):
            val = prec[i] * rec[i]
            if val > max_val:
                max_val = val
                max_thold = tholds[i]
                max_prec = prec[i]
                max_rec = rec[i]
        return max_prec, max_rec, max_thold

    def save_preds(self, probs, ids, fold, pred_f, dset):
        """Save predictions to a specified file.
        probs: array of binary predictions; shape=(2, <num_instances>).
        ids: list of identifiers for the data instances.
        pred_f: folder to save predictions to.
        dset: dataset (e.g. 'train', 'val', 'test')."""
        columns = ['com_id', 'ind_pred']
        fname = dset + '_' + fold + '_preds.csv'

        preds = list(zip(ids, probs[:, 1]))
        preds_df = pd.DataFrame(preds, columns=columns)
        preds_df.to_csv(pred_f + fname, index=None)

    def set_plot_rc(self):
        """Corrects for embedded fonts for text in plots."""
        plt.rc('pdf', fonttype=42)
        plt.rc('ps', fonttype=42)

    def print_median_mean(self, ids, probs, y, fw=None):
        """Prints the median and mean independent predictions for spam and ham.
        ids: comment ids.
        probs: independent predictions.
        y: labels"""
        preds = list(zip(ids, probs[:, 1], y))
        df = pd.DataFrame(preds, columns=['com_id', 'ind_pred', 'label'])
        spam_med = df[df['label'] == 1]['ind_pred'].median()
        ham_med = df[df['label'] == 0]['ind_pred'].median()
        spam_mean = df[df['label'] == 1]['ind_pred'].mean()
        ham_mean = df[df['label'] == 0]['ind_pred'].mean()
        s1 = '\tmedian spam: %.4f, ham: %.4f' % (spam_med, ham_med)
        s2 = '\tmean spam: %.4f, ham: %.4f' % (spam_mean, ham_mean)
        self.write(s1, fw=fw)
        self.write(s2, fw=fw)

    def print_scores(self, max_p, max_r, thold, aupr, auroc, fw=None):
        """Print evaluation metrics to std out.
        max_p: maximum precision in pr curve at thold.
        max_r: maximum recall in pr curve at thold.
        thold: threshold where the maximum area is.
        aupr: area under the pr curve.
        auroc: area under the roc curve."""
        s = '\tmax p: %.3f, max r: %.3f, area: %.3f, thold: %.3f'
        self.write(s % (max_p, max_r, max_p * max_r, thold), fw=fw)
        self.write('\taupr: %.4f, auroc: %.4f' % (aupr, auroc), fw=fw)
