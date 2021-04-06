
import pandas as pd



def load_pima():
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pd.read_csv('./datasets/pima-indians-diabetes.csv', names=names)
    array = dataframe.values
    X = array[:, 0:8]
    Y = array[:, 8]
    return X, Y
X, Y = load_pima()


# create train - test - split of data
# check prediction on validation set, and record [accuracy, depth]
# plot that stuff.


from sklearn.metrics import accuracy_score
from sklearn import tree

from sklearn.tree import export_graphviz
from graphviz import Source
from matplotlib import pyplot as plt

clf = tree.DecisionTreeClassifier(max_depth=3)

clf = clf.fit(X, Y)

predicted_classes = clf.predict(X)
accuracy = accuracy_score(Y.flatten(), predicted_classes)
print(accuracy * 100)

# text_representation = tree.export_text(clf)
# print(text_representation)
#
# graph = Source(export_graphviz(clf, out_file=None))
# graph.format = 'png'
# graph.render('dt', view=True)
