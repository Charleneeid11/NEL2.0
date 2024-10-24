from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

class RandomForestModel:
    def __init__(self, params, data, task="classification"):
        self.task = task
        self.params = params
        self.data = data
        if self.task == "classification":
            self.model = RandomForestClassifier(**self.params)
        else:
            self.model = RandomForestRegressor(**self.params)
    
    def train(self):
        self.model.fit(self.data.x_train, self.data.y_train)
    
    def test(self):
        predictions = self.model.predict(self.data.x_test)
        if self.task == "classification":
            accuracy = accuracy_score(self.data.y_test, predictions)
            print(f"Test Accuracy: {accuracy}")
            return accuracy
        else:
            mse = mean_squared_error(self.data.y_test, predictions)
            print(f"Test MSE: {mse}")
            return mse
    
    def train_and_test(self):
        self.train()
        return self.test()
    
    def set_params(self, new_params):
        self.model.set_params(**new_params)

    def evaluate(self):
        return self.model.score(self.data.x_test, self.data.y_test)
