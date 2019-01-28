"""
This script uses EGGS to model a spam dataset.
"""
import numpy as np
from . import print_utils
from .sgl import SGL
from .joint import Joint
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class EGGS:
    """
    Extended Group-Based Graphical Models for Spam (EGGS).
    """

    def __init__(self, target_col=None, estimator=None, sgl_method=None, stacks=2, joint_model=None,
                 relations=None, sgl_func=None, pgm_func=None, verbose=0):
        """
        Initialization of EGGS classifier.

        Parameters
        ----------
        target_col : str, (default=None)
            Name of the attribute to be predicted.
        sgl_method : str {'cv', 'holdout', None} (default=None)
            If not None, method for stacked graphical learning. Cross-validation (cv) or sequential (holdout).
        stacks : int (default=2)
            Number of stacks to use for SGL. Only relevant if sgl is not None.
        joint_model : str {'psl', 'mrf', None}
            Probabilistic graphical model to use for joint inference.
        relations : list (default=None)
            Relations to use for relational modeling.
        sgl_func : func (default=None)
            Domain-dependent helper method to generate pseudo-relational features.
        pgm_func : func (default=None)
            Domain-dependent helper method to generate relational files for joint inference.
        verbose : int (default=1)
            Prints debugging information, higher outputs the higher verbose is.
        """
        self.estimator = estimator
        self.sgl_method = sgl_method
        self.stacks = stacks
        self.joint_model = joint_model
        self.relations = relations
        self.sgl_func = sgl_func
        self.pgm_func = pgm_func
        self.verbose = verbose

        if estimator is None:
            self.estimator = LogisticRegression()

    def fit(self, X, y, target_col):
        X, y = check_X_y(X, y)
        if y.dtype == np.float and not np.all(np.mod(y, 1) == 0):
            raise ValueError('Unknown label type: ')
        self.n_feats_ = X.shape[1]
        self.classes_ = np.unique(y)

        if self.sgl_method is not None:
            sgl = SGL(self.estimator, self.sgl_func, self.relations, self.sgl_method, stacks=self.stacks)
            self.sgl_ = sgl.fit(X, y, target_col)
        else:
            self.clf_ = clone(self.estimator).fit(X, y)

        return self

    def predict_proba(self, X, target_col):
        X = check_array(X)

        if self.sgl_method is not None:
            check_is_fitted(self, 'sgl_')
            y_hat = self.sgl_.predict_proba(X, target_col)

        else:
            check_is_fitted(self, 'clf_')
            assert hasattr(self.clf_, 'predict_proba')
            y_hat = self.clf_.predict_proba(X)

        if self.joint_model is not None:
            self.joint_ = Joint(self.relations, self.pgm_func, pgm_type=self.joint_model)
            y_hat = self.joint_.inference(y_hat[:, 1], target_col)

        return y_hat

    def predict(self, X, target_col):
        y_score = self.predict_proba(X, target_col)
        return self.classes_[np.argmax(y_score, axis=1)]

    def plot_feature_importance(self, X_cols):

        if self.sgl_method is not None:
            self.sgl_.plot_feature_importance(X_cols + self.sgl_.Xr_cols_)

        else:
            print_utils.print_model(self.clf_, X_cols)
