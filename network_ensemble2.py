from random_forest import RandomForestModel
from random_forest_params_generator import RandomForestParamsGenerator
from data import DataWrapper
from scipy import stats

class Ensemble:
    def __init__(self, data: DataWrapper, number_of_models, task="classification"):
        self.data = data
        self.random_forest_params_generator = RandomForestParamsGenerator()
        self.number_of_models = number_of_models
        self.task = task
        self.models_data = []
        self.models = []
        self.allocate_models_data()
        self.generate_models()

    def allocate_models_data(self):
        for i in range(self.number_of_models):
            self.models_data.append(DataWrapper(self.data, self.data, test_size=0.2, val_size=0.2))

    def generate_models(self):
        for i in range(self.number_of_models):
            params = self.random_forest_params_generator.get_random_params()
            self.models.append(RandomForestModel(params, self.models_data[i], task=self.task))

    def train_ind(self, model_index):
        self.models[model_index].train_and_test()

    def vectorize_model(self, model_index):
        # Implement vectorization logic for Random Forests if necessary
        pass

    def integrate_ensemble(self):
        predictions = []
        for model in self.models:
            predictions.append(model.model.predict(self.data.x_test))
        
        # Perform majority voting or averaging
        if self.task == "classification":
            final_prediction = stats.mode(predictions, axis=0)[0]
        else:
            final_prediction = np.mean(predictions, axis=0)
        
        print(f"Final Ensemble Prediction: {final_prediction}")
        return final_prediction
