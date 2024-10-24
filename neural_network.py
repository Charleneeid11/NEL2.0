import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import utils.main as utils
from data import DataWrapper
import tensorflow_addons as tfa

class NeuralNetworkParams:
    def __init__(self, number_of_hidden_layers, number_of_neurons_in_hidden_layers, hidden_layers_activation_function, output_layer_activation_function, learning_rate):
        self.number_of_hidden_layers = number_of_hidden_layers
        self.number_of_neurons_in_hidden_layers = number_of_neurons_in_hidden_layers
        self.hidden_layers_activation_function = hidden_layers_activation_function
        self.output_layer_activation_function = output_layer_activation_function
        self.learning_rate = learning_rate
    

class NeuralNetwork:
    def __init__(self, params: NeuralNetworkParams, data: DataWrapper):
        self.params = params
        self.data = data
        self.model = self.generate_model(params, data)

    def get_params(self):
        return self.params
        
    def generate_model(self, params: NeuralNetworkParams, data: DataWrapper):
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.InputLayer(input_shape=[data.x_train.shape[1]]))

        for i in range(params.number_of_hidden_layers):
            model.add(
                tf.keras.layers.Dense(
                    params.number_of_neurons_in_hidden_layers,
                    activation=params.hidden_layers_activation_function,
                )
            )
            
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy", tf.keras.metrics.Recall()],
        )
        return model

    def train(self, model_index, epochs=100, verbose=0, draw_accuracy=True):

        print("\n-------------------\nTraining model: ", model_index + 1, "\n-------------------\n")

        print("Data: ", self.data)

        history = self.model.fit(
            self.data.x_train,
            self.data.y_train,
            epochs=epochs,
            verbose=verbose,
            validation_data=(self.data.x_val, self.data.y_val)
        )
        if draw_accuracy:
            utils.plot_accuracy_loss(history, model_index, is_validation_data=True, save_to_folder="./logs/models_accuracy_images")

        print("\nTraining accuracy: ", history.history["accuracy"][-1], "\n")
        print("Training loss: ", history.history["loss"][-1], "\n-------------------\n")

        print("Validation accuracy: ", history.history["val_accuracy"][-1], "\n")
        print("Validation loss: ", history.history["val_loss"][-1], "\n-------------------\n")

        return history
    
    def test(self):
        print("\nEvaluating on test data:...\n")
        results = self.model.evaluate(
        self.data.x_test,
        self.data.y_test,
        verbose=2)
        print("Test loss:", results[0], "\n-------------------\n")
        print("Test accuracy:", results[1], "\n-------------------\n")
        if len(results) > 2:
            print("Test additional metrics:", results[2:], "\n-------------------\n")

        return results
    
    def train_and_test(self, model_index):
        self.train(model_index, epochs=100, verbose=0, draw_accuracy=True)
        self.test()

    def set_weights(self, new_weights):
        self.model.set_weights(new_weights)

    def evaluate(self):
        return self.model.evaluate(self.data.x_test, self.data.y_test, verbose=0)