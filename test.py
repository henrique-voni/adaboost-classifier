from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

from Adaboost import Adaboost

X,y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=40)

# Adaboost com: MLP, ELM, MLP, ELM, MLP e ELM
ada = Adaboost(estimators=["MLP", "ELM"], n_rounds=6, random_state=40)
ada.fit(X_train, y_train)

y_pred = ada.predict(X_test)

print(f"Acurácia: {accuracy_score(y_pred, y_test)}")