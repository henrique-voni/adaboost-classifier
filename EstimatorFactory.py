from sklearn.neural_network import MLPClassifier
from sklearn_extensions.extreme_learning_machines import ELMClassifier

class EstimatorFactory:
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