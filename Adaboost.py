# Sklearn built-in classes for custom estimator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils import check_random_state

# imports
import numpy as np
import itertools

from EstimatorFactory import EstimatorFactory

class Adaboost(BaseEstimator, ClassifierMixin):

    def __init__(self, estimators=["MLP"], n_rounds=5, random_state=None):
        self.estimators = estimators
        self.n_rounds = n_rounds
        self.random_state = random_state
        

    def fit(self, X, y):
        self.random_state_ = check_random_state(self.random_state)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.models_ = []
        
        self.w = np.ones(X.shape[0]) / X.shape[0]

        i = itertools.cycle(self.estimators)
        
        X,y = check_X_y(X, y)

        for _ in range(self.n_rounds):
            
            clf = next(i)
            clf.fit(X,y)
            
            y_pred = clf.predict(X)
            
            clf_err = self.compute_error(y_pred, y)
            clf_alpha = self.compute_alpha(clf_err, self.n_classes_)
            
            self.models_.append((clf, clf_alpha))

            self.w = self.update_weights(self.w)

            X,y = self.resample_with_replacement(X,y)

        return self

    def predict(self, X):
        pass

    def compute_alpha(self, z, K):
        return 0.5 * np.log((1-z) / float(z)) + np.log(K - 1) # Equação adaptada para algoritmo SAMME

    def compute_error(self, y_pred, y):
        miss_w_idx = np.flatnonzero(y_pred != y) # Retorna indices das amostras erradas
        miss_w = np.take(self.w, miss_w_idx) # Retorna os pesos das amostras erradas
        return sum(miss_w) / sum(self.w)

    def update_weights(self,w):
        pass

    def resample_with_replacement(self,X,y):
        pass

    def normalize_weights(self, w):
        return w / sum(w)



    