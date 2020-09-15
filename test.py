from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from Adaboost import Adaboost
import numpy as np


X,y = load_iris(return_X_y=True)


"""
    Teste com holdout
"""
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=40)

# Adaboost com: MLP, ELM, MLP, ELM, MLP e ELM
ada = Adaboost(estimators=["MLP", "ELM"], n_rounds=6, random_state=40)
ada.fit(X_train, y_train)

y_pred = ada.predict(X_test)

print(f"Acurácia: {accuracy_score(y_pred, y_test)}")


"""
    Teste com 5-fold CV
"""
kf = StratifiedKFold(n_splits=5, random_state=40, shuffle=True)
accs = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf = Adaboost(estimators=["MLP", "ELM"], n_rounds=6, random_state=40)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accs.append(accuracy_score(y_pred, y_test))

    print("_____ TERMINOU FOLD ______")

print(accs)
print(f"Média: {np.mean(accs)}")
