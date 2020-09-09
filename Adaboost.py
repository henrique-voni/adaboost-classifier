from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils import check_random_state
from sklearn.neural_network import MLPClassifier
from sklearn_extensions.extreme_learning_machines import ELMClassifier

import numpy as np
import itertools

class EstimatorFactory():
    @staticmethod
    def create(algorithm, params):
       if algorithm == "MLP":
           return EstimatorFactory.create_mlp_(params)
       elif algorithm == "ELM":
           return EstimatorFactory.create_elm_(params)

    @staticmethod
    def create_mlp_(params):
        return MLPClassifier(**params)

    @staticmethod
    def create_elm_(params):
        return ELMClassifier(**params)



class Adaboost(BaseEstimator, ClassifierMixin):

    def __init__(self, estimators=[MLPClassifier(hidden_layer_sizes=10)], n_rounds=5, random_state=None):
        self.estimators = estimators
        self.n_rounds = n_rounds
        self.random_state = random_state
        

    def fit(self, X, y):
        i = itertools.cycle(self.estimators)
        
        self.models_ = []
        self.estimators_ = [next(i) for _ in range(self.n_rounds)]
        self.w = np.ones(X.shape[0]) / X.shape[0]
       
        self.random_state_ = check_random_state(self.random_state)
        X,y = check_X_y(X, y)

        #Primeiro round

        









        return self.estimators

    def predict(self, X):
        pass

    def compute_alpha(self, z):
        return 0.5 * np.log((1-z) / float(z))

    def compute_error(self, y_pred, y):
        miss_w_idx = np.flatnonzero(y_pred != y) # Retorna indices das amostras erradas
        miss_w = np.take(self.w, miss_w_idx) # Retorna os pesos das amostras erradas
        return sum(miss_w) / sum(self.w)


    def normalize_weights(self, w):
        return w / sum(w)



    