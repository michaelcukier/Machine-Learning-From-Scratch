import pandas as pd
import numpy as np

learning_rate = 0.0000000001


# 1) import dataset
def load_pima():
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pd.read_csv('./datasets/pima-indians-diabetes.csv', names=names)
    array = dataframe.values
    X = array[:, 0:8]
    Y = array[:, 8]
    return X, Y
X, Y = load_pima()
# ---------------------------------------------



# 2) train using sk-learn's Logistic Regression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
def sklearn_LR(X, Y):
    model = SGDClassifier(
        max_iter=1000, loss='log', shuffle=False,
        learning_rate='constant', eta0=learning_rate)
    model.fit(X, Y)
    predicted_classes = model.predict(X)
    accuracy = accuracy_score(Y.flatten(), predicted_classes)
    return accuracy*100
accuracy_sk_learn = sklearn_LR(X, Y)
# ---------------------------------------------



# 3) train using my own implementation
class MyOwnLogisticRegression:
    def __init__(self, nb_of_predictors, learning_rate):
        self.bias = 0
        self.weights = [0] * nb_of_predictors
        self.learning_rate = learning_rate

    def logistic(self, val):
        return 1/(1+np.exp(-(val)))

    def train(self, predictors, targets, iter):
        for _ in range(iter):
            for p, t in zip(predictors, targets):
                # calculate prediction
                equation = self.bias
                for idx, feature in enumerate(p):
                    equation += feature * self.weights[idx]
                prediction = self.logistic(equation)
                # update weights given prediction
                self.bias = self.bias + self.learning_rate * (t - prediction) * prediction * (1 - prediction)
                for idx, w in enumerate(self.weights):
                    new_w = w + self.learning_rate * (t - prediction) * prediction * (1 - prediction) * p[idx]
                    self.weights[idx] = new_w

    def predict(self, input):
        prediction_ = self.bias
        for idx, feature in enumerate(input):
            prediction_ += feature * self.weights[idx]
        prediction = self.logistic(prediction_)
        if prediction < 0.5:
            return 0
        else:
            return 1

LR = MyOwnLogisticRegression(
        nb_of_predictors=X.shape[1],
        learning_rate=learning_rate)

LR.train(X, Y, 1000)
num_correct = 0
for x, y in zip(X, Y):
    if LR.predict(x) == int(y):
        num_correct += 1

accuracy_own_implementation = (num_correct/len(X)) * 100
# ---------------------------------------------


# 4) compare
print('Sklearn acc:           ', accuracy_sk_learn)
print('Own implementation acc:', accuracy_own_implementation)


