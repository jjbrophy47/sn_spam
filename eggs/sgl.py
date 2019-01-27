"""
This module contains high-level APIs implementing stacked graphical learning (SGL).
"""
import numpy as np
from . import utils
from . import print_utils
from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class SGL():

    def __init__(self, estimator, pr_func, relations, method='cv', folds=10, stacks=2):
        self.estimator = estimator
        self.pr_func = pr_func
        self.relations = relations
        self.method = method
        self.folds = folds
        self.stacks = stacks

    def fit(self, X, y, target_col):
        """Performs stacked graphical learning using the cross-validation or holdout method."""

        X, y = check_X_y(X, y)

        if self.method == 'cv':
            self._cross_validation(self.estimator, X, y, target_col, self.pr_func, self.relations,
                                   self.folds, self.stacks)
        else:
            self._holdout(self.estimator, X, y, target_col, self.pr_func, self.relations, self.stacks,
                          self.folds, self.stacks)

        return self

    def predict_proba(self, Xg, target_col):

        Xg = check_array(Xg)
        check_is_fitted(self, 'base_model_')
        check_is_fitted(self, 'stacked_models_')

        y_hat = self.base_model_.predict_proba(Xg)[:, 1]
        Xr, Xr_cols = self.pr_func(y_hat, target_col, self.relations)

        for stacked_model in self.stacked_models_:
            X = np.hstack([Xg, Xr])
            y_hat = stacked_model.predict_proba(X)[:, 1]
            Xr, Xr_cols = self.pr_func(y_hat, target_col, self.relations)

        y_score = np.hstack([1 - y_hat.reshape(-1, 1), y_hat.reshape(-1, 1)])
        return y_score

    def plot_feature_importance(self, X_cols):

        last_model = self.stacked_models_[-1]
        print_utils.print_model(last_model, X_cols)

    # private
    def _cross_validation(self, clf, Xg, y, target_col, pr_func, relations, folds, stacks):

        Xr = None
        self.base_model_ = clone(clf).fit(Xg, y)
        self.stacked_models_ = []

        for i in range(stacks):
            X = Xg if i == 0 else np.hstack([Xg, Xr])

            y_hat = utils.cross_val_predict(X, y, clf, num_cvfolds=folds)
            Xr, Xr_cols = pr_func(y_hat, target_col, relations)

            X = np.hstack([Xg, Xr])
            clf = clone(clf).fit(X, y)
            self.stacked_models_.append(clf)

        self.Xr_cols_ = Xr_cols

    def _holdout(clf, X, y, X_cols, target_col, relations, pr_func, stacks=2):
        pass
