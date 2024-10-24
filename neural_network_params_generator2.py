import numpy as np
from random_forest import RandomForestModel

class RandomForestParamsGenerator:
    def __init__(self):
        self.n_estimators = [10, 200]  # Number of trees
        self.max_depth = [5, 50]  # Maximum depth of the trees
        self.min_samples_split = [2, 10]  # Minimum number of samples to split a node
        self.min_samples_leaf = [1, 5]  # Minimum number of samples in a leaf node

    def get_random_params(self):
        return {
            "n_estimators": np.random.randint(self.n_estimators[0], self.n_estimators[1]),
            "max_depth": np.random.randint(self.max_depth[0], self.max_depth[1]),
            "min_samples_split": np.random.randint(self.min_samples_split[0], self.min_samples_split[1]),
            "min_samples_leaf": np.random.randint(self.min_samples_leaf[0], self.min_samples_leaf[1]),
        }
