from network_ensemble import Ensemble
from data import DataWrapper
import pandas as pd
from sklearn.datasets import make_circles
def main():
    # path="C:/Users/charl/Desktop/Documents/LAU/Semesters/Grad/Courses Taken/Fall2023/Topics - Network Science/Final Project/tvmarketing.csv"
    # df = pd.read_csv('your_file.csv')
    # x = df.iloc[:, 0]
    # y = df.iloc[:, 1]
    x, y = make_circles(10000)
    # data = DataWrapper(x, y, 0.2, 0.2)
    # ensemble = Ensemble(data, 10, 100, 0.5, "cosine", "degree")
    # ensemble.train_ind(epochs=100, verbose=0, draw_accuracy=True)
    #ensemble.evaluate_models()
    

if __name__ == "__main__":
    main()