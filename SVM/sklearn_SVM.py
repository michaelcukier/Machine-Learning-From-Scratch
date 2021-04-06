# # Compare Algorithms
# import pandas
# from sklearn import model_selection
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
#
# # load dataset
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#
#
# dataframe = pandas.read_csv(url, names=names)
#
# dataframe[(dataframe['class'] == 0)] = -1
#
# array = dataframe.values
# X = array[:,0:8]
# Y = array[:,8]
#
# import numpy as np
#
#
# class SVM:
#
#     def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
#         self.lr = learning_rate
#         self.lambda_param = lambda_param
#         self.n_iters = n_iters
#         self.w = None
#         self.b = None
#
#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#
#         y_ = np.where(y <= 0, -1, 1)
#
#         self.w = np.zeros(n_features)
#         self.b = 0
#
#         for _ in range(self.n_iters):
#             for idx, x_i in enumerate(X):
#                 condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
#                 if condition:
#                     self.w -= self.lr * (2 * self.lambda_param * self.w)
#                 else:
#                     self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
#                     self.b -= self.lr * y_[idx]
#
#     def predict(self, X):
#         approx = np.dot(X, self.w) - self.b
#         return np.sign(approx)
#
#
# s = SVM()
#
# s.fit(X, Y)
#
# dataframe[(dataframe['class'] == -1)] = 0
#
# array = dataframe.values
# X = array[:,0:8]
# Y = array[:,8]
#
# num_correct = 0
# for x, y in zip(X, Y):
#     if s.predict(x) == int(y):
#         num_correct += 1
#
#
#
# accuracy_own_implementation = (num_correct/len(X)) * 100
#
# print(accuracy_own_implementation)



import pandas
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']


dataframe = pandas.read_csv(url, names=names)

# dataframe[(dataframe['class'] == 0)] = -1

array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# prepare configuration for cross validation test harness
# prepare models
model = SVC()
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

model.fit(X, Y)

predicted_classes = model.predict(X)

# dataframe[(dataframe['class'] == -1)] = 0

array = dataframe.values
X = array[:,0:8]
Y = array[:,8]


accuracy = accuracy_score(Y.flatten(), predicted_classes)


print(accuracy * 100)
#
# # for name, model in models:
# # 	# kfold = model_selection.KFold(n_splits=10)
# # 	# cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# #
# # 	# results.append(cv_results)
# # 	# names.append(name)
# # 	# msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
# # 	print(msg)
#
#

