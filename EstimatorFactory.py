from sklearn.neural_network import MLPClassifier
from sklearn_extensions.extreme_learning_machines import ELMClassifier

class EstimatorFactory:

    @staticmethod
    def create(algorithm, random_state=None, elm_params=None, mlp_params=None):
       if algorithm == "MLP":
           return EstimatorFactory.create_mlp_(params=mlp_params, random_state=random_state)
       elif algorithm == "ELM":
           return EstimatorFactory.create_elm_(params=elm_params, random_state=random_state)

    @staticmethod
    def create_mlp_(params, random_state):
        return MLPClassifier(**params, random_state=random_state)

    @staticmethod
    def create_elm_(params, random_state):
        return ELMClassifier(**params, random_state=random_state)
