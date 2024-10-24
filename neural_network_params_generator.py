from neural_network import NeuralNetwork, NeuralNetworkParams


class NeuralNetworkParamsGenerator:
    def __init__(self):    
        self.number_of_hidden_layers = [1, 4]
        self.number_of_neurons_in_hidden_layers = [10, 100]
        self.hidden_layers_activation_function = [1, 3]
        self.output_layer_activation_function = [1, 2]
        self.learning_rate = [0.00001, 0.1]
        
    def get_random_params(self):
        return NeuralNetworkParams(
            number_of_hidden_layers = np.random.randint(self.number_of_hidden_layers[0], self.number_of_hidden_layers[1]),
            number_of_neurons_in_hidden_layers = np.random.randint(self.number_of_neurons_in_hidden_layers[0], self.number_of_neurons_in_hidden_layers[1]),
            hidden_layers_activation_function = np.random.choice(self.hidden_layers_activation_function),
            output_layer_activation_function = np.random.choice(self.output_layer_activation_function),
            learning_rate = np.random.uniform(self.learning_rate[0], self.learning_rate[1])
        )