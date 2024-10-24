import numpy as np
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from utils.main import check_balance

def load_wine_dataset():
    """
    Load the wine dataset from wine_quality.csv
    :return: a tuple of 2 elements, x and y
    """
    data = np.genfromtxt('datasets/wine_quality.csv', delimiter=',', dtype=None, encoding=None)

    data = data[1:] # remove the header
    
    x = []
    y = []

    for row in data:
        x_entries = [row[i] for i in range(len(row)) if i != len(row)-1]
        y_entries = row[-1]

        x.append(np.array(x_entries))
        y.append(y_entries)
    
    x = np.array(x)
    y = np.array(y)

    return x, y

def load_diabetes_dataset():
    """
    Load the diabetes dataset from sklearn.
    :return: a tuple of 2 elements, x and y
    """
    data = np.genfromtxt('datasets/diabetes_data.csv', delimiter=',', dtype=None, encoding=None)

    data = data[1:] # remove the header
    
    x = []
    y = []

    for row in data:
        x_entries = [row[i] for i in range(len(row)) if i != 0]
        y_entries = row[0]

        x.append(np.array(x_entries))
        y.append(y_entries)
    
    x = np.array(x)
    y = np.array(y)
    
    return x, y

def load_smoke_detection_dataset():
    """
    Load the smoke detection dataset from sklearn.
    :return: a tuple of 2 elements, x and y
    """
    data = np.genfromtxt('datasets/smoke_detection_iot.csv', delimiter=',', dtype=None, encoding=None, skip_header=1)
    
    x = []
    y = []

    for row in data:
        x_entries = [row[i] for i in range(len(row)) if i >= 2 and i != len(row)-1]
        y_entries = row[-1]

        x.append(np.array(x_entries))
        y.append(y_entries)
    
    x = np.array(x)
    y = np.array(y)
    check_balance(y)
    return x, y

def load_breast_cancer_dataset():
    """
    Load the breast cancer dataset from sklearn.
    :return: a tuple of 2 elements, x and y
    """
    x, y = load_breast_cancer(return_X_y=True)
    check_balance(y)
    return x, y


def load_iris_dataset():
    """
    Load the iris dataset from sklearn.
    :return: a tuple of 2 elements, x and y
    """
    x, y = load_iris(return_X_y=True)
    return x, y

def load_magic_gamma_dataset():
    """
    Load the magic gamma dataset from sklearn.
    :return: a tuple of 2 elements, x and y
    """
    
    data = np.genfromtxt('datasets/magic04.data', delimiter=',', dtype=None, encoding=None)
    x = []
    y = []

    for row in data:
        x_entries = [row[i] for i in range(len(row)) if i != len(row)-1]
        y_entries = row[-1]

        x.append(np.array(x_entries))
        y.append(y_entries)

    x = np.array(x)
    y = np.array(y)
    
    return x, y

def load_star():
    """
    Load the magic gamma dataset from sklearn.
    :return: a tuple of 2 elements, x and y
    """
    
    data = np.genfromtxt('datasets/Star39552_balanced.csv', delimiter=',', dtype=None, encoding=None)
    x = []
    y = []

    label_encoder = LabelEncoder()
    for row in data:
        x_entries = [row[i] for i in range(len(row)) if i != len(row)-1]
        y_entries = row[-1]
        x_entries[4] = label_encoder.fit_transform([x_entries[4]])[0]

        x.append(np.array(x_entries))
        y.append(y_entries)

    x = np.array(x)
    y = np.array(y)
    
    check_balance(y)
    print(x.shape)
    return x, y


def load_electricity_dataset():
    """
    Load the electricity dataset.
    :return: a tuple of 2 elements, x and y
    """
    
    data = np.genfromtxt('datasets/electricity.data', delimiter=',', dtype=None, encoding=None)
    x = []
    y = []

    for row in data:
        x_entries = [row[i] for i in range(len(row)) if i != len(row)-1]
        y_entries = row[-1]

        x.append(np.array(x_entries))
        y.append(y_entries)

    x = np.array(x)
    y = np.array(y)
    
    return x, y

def load_eeg_eye_state_dataset():
    """
    Load the EEG EYE STATE dataset.
    :return: a tuple of 2 elements, x and y
    """
    
    data = np.genfromtxt('datasets/EEGEyeState.data', delimiter=',', dtype=None, encoding=None)
    x = []
    y = []

    for row in data:
        x_entries = [row[i] for i in range(len(row)) if i != len(row)-1]
        y_entries = row[-1]

        x.append(np.array(x_entries))
        y.append(y_entries)

    x = np.array(x)
    y = np.array(y)

    check_balance(y)
    return x, y

def load_dry_beans():
    data = np.array(pd.read_excel('datasets/Dry_Bean_Dataset.xlsx'))
    
    data = data[1:]
    x = []
    y = []

    for row in data:
        x_entries = [row[i] for i in range(len(row)) if i != len(row)-1]
        y_entries = row[-1]

        x.append(np.array(x_entries))
        y.append(y_entries)
    
    x = np.array(x)
    y = np.array(y)
    
    
    check_balance(y)

    return x, y

def load_htru():
    data =  np.genfromtxt('datasets/HTRU_2.csv', delimiter=',', dtype=None, encoding=None)
    
    x = []
    y = []

    for row in data:
        x_entries = [row[i] for i in range(len(row)) if i != len(row)-1]
        y_entries = row[-1]

        x.append(np.array(x_entries))
        y.append(y_entries)
    
    x = np.array(x)
    y = np.array(y)
    
    print(x.shape)
    check_balance(y)

    return x, y