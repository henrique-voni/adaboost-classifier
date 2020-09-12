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
    def create(algorithm, random_state):
       if algorithm == "MLP":
           return EstimatorFactory.create_mlp_(params=EstimatorFactory.MLP_PARAMS, random_state=random_state)
       elif algorithm == "ELM":
           return EstimatorFactory.create_elm_(params=EstimatorFactory.ELM_PARAMS, random_state=random_state)

    @staticmethod
    def create_mlp_(params, random_state):
        return MLPClassifier(**params, random_state=random_state)

    @staticmethod
    def create_elm_(params, random_state):
        return ELMClassifier(**params, random_state=random_state)