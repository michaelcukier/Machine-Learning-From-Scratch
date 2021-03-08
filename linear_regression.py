import pandas as pd
import numpy as np

learning_rate = 0.0000000001


# 1) import dataset
def load_housing():
    dataframe = pd.read_fwf('./datasets/housing.data')
    array = dataframe.values
    X = array[:, 0:13]
    Y = array[:, 13]
    return X, Y
X, Y = load_housing()
# ---------------------------------------------




# 2) train using sk-learn
from sklearn.linear_model import SGDRegressor
model = SGDRegressor(max_iter=1000, shuffle=False,
        learning_rate='constant', eta0=learning_rate)
model.fit(X, Y)
# ---------------------------------------------




# 3) train using my own implementation
class MyOwnLinearRegression:
    def __init__(self, nb_of_predictors, learning_rate):
        self.bias = 0
        self.weights = [0] * nb_of_predictors
        self.learning_rate = learning_rate

    def train(self, predictors, targets, iter):
        for _ in range(iter):
            for p, t in zip(predictors, targets):

                # make prediction
                prediction = self.bias
                for idx_i, input_data in enumerate(p):
                    prediction += self.weights[idx_i] * input_data

                # calc error
                error = prediction - t

                # update bias & weights
                self.bias = self.bias - self.learning_rate * error
                for idx_w, w in enumerate(self.weights):
                    new_w = w - self.learning_rate * error * p[idx_w]
                    self.weights[idx_w] = new_w

    def predict(self, input):
        prediction__ = self.bias
        for idx, feature in enumerate(input):
            prediction__ += feature * self.weights[idx]
        return prediction__


LR = MyOwnLinearRegression(
    nb_of_predictors=X.shape[1],
    learning_rate=learning_rate)

LR.train(X, Y, 1000)
# ---------------------------------------------




# 4) compare
single_input_data_test = [0.00632, 18.00, 2.310, 0, 0.5380,
                   6.5750, 65.20, 4.0900, 1, 296.0,
                   15.30, 396.90, 4.98]

mine = LR.predict(single_input_data_test)
sk = model.predict([ single_input_data_test ])

print('My prediction:', mine)
print('SK prediction:', sk[0])

