class MySVM:
    def __init__(self, X, Y, learning_rate, iterations):
        self.predictors = X
        self.targets = Y
        self.coefs = [0.0] * X.shape[1]
        self.learning_rate = learning_rate
        self.iterations = iterations

    def train(self):
        it = 0
        for _ in range(self.iterations):
            for p, t in zip(self.predictors, self.targets):
                it += 1

                # make a prediction
                out = 0
                for idx, feature in enumerate(p):
                    out += feature * self.coefs[idx]
                out *= t

                # update parameters depending on output
                if out > 1:
                    for idx, c in enumerate(self.coefs):
                        self.coefs[idx] = (1 - (1/it)) * self.coefs[idx]
                else:
                    for idx, (a, b) in enumerate(zip(self.coefs, p)):
                        self.coefs[idx] = ((1 - (1/it)) * self.coefs[idx]) + (1/(self.learning_rate * it)) * (t * b)

    def predict(self, inp):
        prediction_ = 0
        for idx, feature in enumerate(inp):
            prediction_ += feature * self.coefs[idx]
        if prediction_ < 0:
            return 0
        else:
            return 1

import pandas as pd

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('../datasets/pima-indians-diabetes.csv', names=names)

# data[(data['class'] == 0)] = -1

array = data.values
X = array[:, 0:8]
Y = array[:, 8]

svm = MySVM(
    X=X,
    Y=Y,
    learning_rate=0.002,
    iterations=1000
)

svm.train()


num_correct = 0
for x, y in zip(X, Y):
    if svm.predict(x) == int(y):
        num_correct += 1

accuracy_own_implementation = (num_correct/len(X)) * 100

print(accuracy_own_implementation)