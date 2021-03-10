import pandas as pd
import numpy as np

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']


# 1) import dataset
def load_pima():
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pd.read_csv('./datasets/pima-indians-diabetes.csv', names=names)
    return dataframe
dataset = load_pima()

# from sklearn import preprocessing
#
# min_max_scaler = preprocessing.MinMaxScaler()
# np_scaled = min_max_scaler.fit_transform(dataset)
# dataset = pd.DataFrame(np_scaled, columns = names)

#
# x = dataset.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# dataset = pd.DataFrame(x_scaled)
# dataset.columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# # ---------------------------------------------
#
# print(dataset.describe())
#
#
# quit()


# 2) train using sk-learn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

array = dataset.values
X = array[:, 0:8]
Y = array[:, 8]

model = LinearDiscriminantAnalysis()
model.fit(X, Y)
predicted_classes = model.predict(X)
accuracy = accuracy_score(Y.flatten(), predicted_classes)
# ---------------------------------------------





# 3) train using my own implementation
class MyOwnLinearDiscrimantAnalysis:
    def __init__(self, dataset):
        self.dataset = dataset

        self.c0_means = None
        self.c1_means = None

        self.c0_var = None
        self.c1_var = None

        self.c0_prob = None
        self.c1_prob = None

    def get_means(self):
        predictors = self.dataset

        predictors_c0 = predictors.loc[predictors['class'] == 0]
        predictors_c0 = predictors_c0[predictors_c0.columns[:-1]]
        predictors_c0_means = predictors_c0.mean()
        predictors_c0_means_dict = predictors_c0_means.to_dict()

        predictors_c1 = predictors.loc[predictors['class'] == 1]
        predictors_c1 = predictors_c1[predictors_c1.columns[:-1]]
        predictors_c1_means = predictors_c1.mean()
        predictors_c1_means_dict = predictors_c1_means.to_dict()

        self.c0_means = list(predictors_c0_means_dict.values())
        self.c1_means = list(predictors_c1_means_dict.values())

    def get_variance(self):
        predictors = self.dataset

        predictors_c0 = predictors.loc[predictors['class'] == 0]
        predictors_c0 = predictors_c0[predictors_c0.columns[:-1]]
        predictors_c0_var = predictors_c0.var()
        predictors_c0_var_dict = predictors_c0_var.to_dict()

        predictors_c1 = predictors.loc[predictors['class'] == 1]
        predictors_c1 = predictors_c1[predictors_c1.columns[:-1]]
        predictors_c1_var = predictors_c1.var()
        predictors_c1_var_dict = predictors_c1_var.to_dict()

        self.c0_var = list(predictors_c0_var_dict.values())
        self.c1_var = list(predictors_c1_var_dict.values())

    def get_class_prob(self):
        classes_prob = self.dataset.groupby('class').size().div(len(self.dataset))
        self.c0_prob = classes_prob.to_dict()[0]
        self.c1_prob = classes_prob.to_dict()[1]

    def predict(self, input):
        discriminants = [0, 0]

        for idx, p in enumerate(input):
            discriminants[0] += (p * (self.c0_means[idx]/np.power(self.c0_var[idx], 2))) - (np.power(self.c0_means[idx], 2)/(2*self.c0_var[idx])) + np.log(self.c0_prob)
        # discriminants[0] += np.log(self.c0_prob)

        for idx, p in enumerate(input):
            discriminants[1] += (p * (self.c1_means[idx]/np.power(self.c1_var[idx], 2))) - (np.power(self.c1_means[idx], 2)/(2*self.c1_var[idx])) + np.log(self.c1_prob)
        # discriminants[1] += np.log(self.c1_prob)

        return 0 if discriminants[0] > discriminants[1] else 1


LDA = MyOwnLinearDiscrimantAnalysis(dataset)
LDA.get_variance()
LDA.get_means()
LDA.get_class_prob()


num_correct = 0
for x, y in zip(X, Y):
    print(LDA.predict(x))
    if LDA.predict(x) == int(y):
        num_correct += 1
accuracy_own_implementation = (num_correct/len(X)) * 100

print('Own implementation:', accuracy_own_implementation)
print('Sklearn accuracy  :', accuracy*100)


