from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils import check_random_state
import numpy as np


class Adaboost(BaseEstimator, ClassifierMixin):

    def __init__(self, estimators=["MLP"], n_rounds=5, random_state=None):
        self.estimators = estimators
        self.n_rounds = n_rounds
        self.random_state = random_state
        

    def fit(self, X, y):
        self.w = np.ones(X.shape[0]) / X.shape[0]
        self.random_state_ = check_random_state(self.random_state)
        X,y = check_X_y(X, y)
        return self

    def predict(self, X):
        pass

    def compute_alpha(self, z):
        return 0.5 * np.log((1-z) / float(z))

    def compute_error(self, y_pred, y):
        # retornar quais indices tb
        return len(sample for (y_pd, y_real) in zip(y_pred, y) if y_pd != y_real)


    def normalize_weights(self, w):
        return w / sum(w)



    