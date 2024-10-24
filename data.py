from time import sleep
# import utils.main
import sklearn
import tensorflow
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
# from utils.main import show_data_set
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer

class DataWrapper:
    def __init__(self, x, y, test_size, val_size):
        self.x, self.y = x, y
        self.test_size = test_size
        self.val_size = val_size*(1 - test_size)  # Adjust validation size based on remaining data after test split
        
        scaler = StandardScaler()
        normalizer = Normalizer()

        # Splitting the data into train+validation and test
        x_temp, self.x_test, y_temp, self.y_test = train_test_split(
            self.x, self.y, test_size=self.test_size, shuffle=True, stratify=None if len(self.y.shape) == 1 else self.y)
        
        # Splitting the train+validation data into train and validation
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_temp, y_temp, test_size=self.val_size, shuffle=True, stratify=None if len(y_temp.shape) == 1 else y_temp)
        
        # Preprocessing features for train, validation, and test data
        self.x_train = self.preprocess_features(self.x_train, scaler, normalizer, fit=True)
        self.x_val = self.preprocess_features(self.x_val, scaler, normalizer, fit=False)
        self.x_test = self.preprocess_features(self.x_test, scaler, normalizer, fit=False)
        
        # Reshaping target variables
        self.y_train = self.y_train.astype('float32').reshape(-1, 1)
        self.y_val = self.y_val.astype('float32').reshape(-1, 1)
        self.y_test = self.y_test.astype('float32').reshape(-1, 1)

    def preprocess_features(self, x, scaler, normalizer, fit=False):
        x = x.astype('float32')
        if fit:
            x = scaler.fit_transform(x)
            x = normalizer.fit_transform(x)
        else:
            x = scaler.transform(x)
            x = normalizer.transform(x)
        return x

    def __str__(self):
        return f"""Data Preview:
        x_train shape: {self.x_train.shape}
        y_train shape: {self.y_train.shape}
        x_val shape: {self.x_val.shape}
        y_val shape: {self.y_val.shape}
        x_test shape: {self.x_test.shape}
        y_test shape: {self.y_test.shape}
        First x_train sample: {self.x_train[0]}
        First y_train sample: {self.y_train[0]}
        -----------------------------"""