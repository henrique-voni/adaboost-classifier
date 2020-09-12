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
        # np.random.seed(self.random_state_)


        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.models = []
        self.n_samples = X.shape[0]
        
        self.w = np.ones(self.n_samples) / self.n_samples

        i = itertools.cycle(self.estimators)
        
        X,y = check_X_y(X, y)

        for _ in range(self.n_rounds):
            
            clf = EstimatorFactory.create(next(i))
            clf.fit(X,y)

            y_pred = clf.predict(X)

            clf_err = self.compute_error(y_pred, y)

            # Condição para SAMME: (1 - erro) < 1/n_classes
            if(1 - clf_err) < (1/self.n_classes):
                raise ValueError("Weak classifer performed poorly, so (1-err) < (1 / n_classes).")


            clf_alpha = self.compute_alpha(clf_err)
            
            self.models.append((clf, clf_alpha))
            self.update_weights(y_pred, y, clf_alpha)

            X,y = self.resample_with_replacement(X,y)
            print("Classifier fitted successfully")

        return self

    def predict(self, X):
        
        predict_matrix = np.zeros( (len(self.models), self.n_samples) )

        predicted_values = np.zeros( (self.n_samples, self.n_classes))

        for i, (clf, alpha) in enumerate(self.models):
            clf_pred = clf.predict(X)
            predict_matrix[i] = clf_pred
        
        # Transpor a matriz para que cada coluna seja a predição de um classificador
        predict_matrix = predict_matrix.T

        return predict_matrix





    def compute_alpha(self, z):
        return 0.5 * np.log((1-z) / float(z)) + np.log(self.n_classes - 1) # Equação adaptada para algoritmo SAMME

    def compute_error(self, y_pred, y):
        miss_w_idx = np.flatnonzero(y_pred != y) # Retorna indices das amostras erradas
        miss_w = np.take(self.w, miss_w_idx) # Retorna os pesos das amostras erradas
        print("Error: ", sum(miss_w) / sum(self.w))
        return sum(miss_w) / sum(self.w)

    def update_weights(self, y_pred, y, alpha):
        ## atualizar apenas amostras classificadas incorretamente
        wrong_idx = np.nonzero(y_pred != y)
        wrong_sample_weights = np.take(self.w, wrong_idx)
        wrong_sample_weights = wrong_sample_weights * np.exp(alpha)
        np.put(self.w, wrong_idx, wrong_sample_weights)
        self.normalize_weights()

    def resample_with_replacement(self, X, y):   
        y = y.reshape(y.shape[0], 1)
        X = np.append(X,y, axis=1)
        resampled_idx = np.random.choice(X.shape[0], X.shape[0], p=self.w)  
        resamples = np.take(X, resampled_idx, axis=0)

        new_y = resamples[:, -1]
        new_X = np.delete(resamples, -1, 1)
        return new_X, new_y    


    def normalize_weights(self):
        self.w = self.w / sum(self.w)


from sklearn.datasets import load_iris

ada = Adaboost()
X,y = load_iris(return_X_y=True)

ada.fit(X,y)


y_pred = ada.predict(X)
print(y_pred)