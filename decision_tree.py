#
# x_1 = [2.771244718,
#        1.728571309,
#        3.678319846,
#        3.961043357,
#        2.999208922,
#        7.497545867,
#        9.00220326,
#        7.444542326,
#        10.12493903,
#        6.642287351]
#
# x_2 = [1.784783929,
#        1.169761413,
#        2.81281357,
#        2.61995032,
#        2.209014212,
#        3.162953546,
#        3.339047188,
#        0.476683375,
#        3.234550982,
#        3.319983761]
#
# Y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
#
#
#
# # let's compute all the gini indexes for the 1st candidate split point
#

import pandas as pd

def load_pima():
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pd.read_csv('./datasets/pima-indians-diabetes.csv', names=names)
    array = dataframe.values
    X = array[:, 0:8]
    Y = array[:, 8]
    return X, Y
X, Y = load_pima()



from sklearn import tree
clf = tree.DecisionTreeClassifier()

clf = clf.fit(X, Y)



tree.plot_tree(clf)














