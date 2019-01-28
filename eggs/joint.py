"""
This module contains high-level APIs implementing FOL joint inference.
"""
from .mrf import MRF
from .psl import PSL
from sklearn.utils.validation import check_array, check_is_fitted


class Joint:
    """
    High-level class with multiple pgm implementations.
    """

    def __init__(self, relations, relations_func, pgm_type='psl', working_dir='.temp/'):
        """
        Initialization of joint inference class.

        Parameters
        ----------
        relations : list (default=None)
            Relations to use for relational modeling.
        relations_func : func (default=None)
            Domain-dependent helper method to generate pgm files.
        pgm_type : str (default='psl') {'psl', 'mrf'}
            Type of PGM to use for joint inference.
        working_dir : str (default='.temp/')
            Temporary directory to store inermdiate files.
        """
        self.relations = relations
        self.relations_func = relations_func
        self.pgm_type = pgm_type
        self.working_dir = working_dir

    def inference(self, y_hat, target_col):
        """Performs joint inference."""

        assert len(y_hat) == len(target_col)

        if self.pgm_type == 'psl':
            pgm = PSL(self.relations, self.relations_func, self.working_dir)
        else:
            pgm = MRF(self.relations, self.relations_func, self.working_dir)

        y_hat = pgm.inference(y_hat, target_col)

        return y_hat

    def predict_proba(self, X, target_col):

        X = check_array(X)
        check_is_fitted(self, 'base_model_')
