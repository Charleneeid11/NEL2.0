from neural_network import NeuralNetwork
from neural_network_params_generator import NeuralNetworkParamsGenerator
from data import DataWrapper
from SimType import SimType
from CentType import CentType
from sklearn.metrics.pairwise import cosine_similarity


# multi-threading and multi-processing
import threading
import multiprocessing
import time



class Ensemble:
    def __init__(self, data: DataWrapper, number_of_models, epoch: int, edge_prob: float, sim_type: SimType, cent_type : CentType):
        if not 50 <= epoch <= 200:
            raise ValueError("Epoch value must be between 50 and 200 inclusive.")
        if not 0.2 <= edge_prob <= 1:
            raise ValueError("Edge_prob value must be between 0.2 and 1 inclusive.")
        self.data = data
        self.neural_network_params_generator = NeuralNetworkParamsGenerator()
        self.number_of_models = number_of_models
        self.epoch = epoch
        self.edge_prob = edge_prob
        self.sim_type = sim_type
        self.cent_type = cent_type
        self.models_data = []
        self.models = []
        self.allocate_models_data()
        self.generate_models()

    def allocate_models_data(self):
        for i in range(self.number_of_models):
            self.models_data.append(DataWrapper(self.data, self.data, test_size=0.2, val_size=0.2))
        
    def generate_models(self):
        for i in range(self.number_of_models):
            self.models.append(NeuralNetwork(self.neural_network_params_generator.get_random_params(), self.models_data[i]))

    def train_ind(self, model_index):
        # multi-processing
        # run each model in a separate process in parallel
        processes = []
        for i in range(self.number_of_models):
            processes.append(multiprocessing.Process(target=self.models[i].train_and_test, args=(model_index)))
            processes[i].start()
        for i in range(self.number_of_models):
            processes[i].join()

    def vectorize_model(self, i):
        norm_hidden_layers=((self.models[i].number_of_hidden_layers)-1)/3
        norm_neurons=((self.models[i].number_of_neurons_in_hidden_layers)-10)/90
        norm_hlaf=((self.models[i].hidden_layers_activation_function)-1)/2
        norm_olaf=((self.models[i].output_layer_activation_function)-1)
        norm_lr=((self.models[i].learning_rate)-0.00001)/0.09999
        normalized_params=[]
        normalized_params.extend(norm_hidden_layers, norm_neurons, norm_hlaf, norm_olaf, norm_lr)
        return normalized_params

    def cosine_similarity(self, i, j):
        v1=self.vectorize_model(i)
        v2=self.vectorize_model(j)
        similarity_matrix=cosine_similarity(v1, v2)
        return similarity_matrix
        
    def evaluate_models(self):
        loss, accuracy = [], []
        
        processes = []
        for i in range(self.number_of_models):
            processes.append(multiprocessing.Process(target=self.models[i].evaluate))
            processes[i].start()
        for i in range(self.number_of_models):
            processes[i].join()
            
        
        # get loss and accuracy from each model
        for i in range(self.number_of_models):
            loss.append(processes[i].loss)
            accuracy.append(processes[i].accuracy)

    
