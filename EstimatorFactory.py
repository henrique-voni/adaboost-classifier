from sklearn.neural_network import MLPClassifier
from sklearn_extensions.extreme_learning_machines import ELMClassifier

class EstimatorFactory:
    
    ### Default parameters for each classifier
    MLP_PARAMS = {
        "hidden_layer_sizes" : 10, 
        "solver": "adam", 
        "max_iter" : 2000, 
        "activation" : "relu"
    }

    ELM_PARAMS = {
        "n_hidden" : 30
    }

    @staticmethod
    def create(algorithm):
       if algorithm == "MLP":
           return EstimatorFactory.create_mlp_(params=EstimatorFactory.MLP_PARAMS)
       elif algorithm == "ELM":
           return EstimatorFactory.create_elm_(params=EstimatorFactory.ELM_PARAMS)

    @staticmethod
    def create_mlp_(params):
        return MLPClassifier(**params)

    @staticmethod
    def create_elm_(params):
        return ELMClassifier(**params)