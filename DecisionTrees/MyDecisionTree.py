import pandas as pd

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('../datasets/pima-indians-diabetes.csv', names=names)


# -----------

'''

How to code a decision tree algorithm?

1) Calculate gini impurity for dataset.
2) From that, calculate information gain for each feature in the dataset.
3) Use highest information gain to find the split feature.
4) Partition the dataset using the feature found in 3).
5) Do 1)-4) for how many levels of depth we want our tree to be.


'''


# gini, IG and best split feature finding


def calc_gini(data):
    class_0_count = data[data['class'] == 0].shape[0]
    class_1_count = data[data['class'] == 1].shape[0]
    dataset_count = data.shape[0]
    p_0 = class_0_count / dataset_count
    p_1 = class_1_count / dataset_count
    return (p_0 * (1 - p_0)) + (p_1 * (1 - p_1))

def partition(data, feature_idx, feature):
    left_side = data[data[names[feature_idx]] <= feature]
    right_side = data[data[names[feature_idx]] > feature]
    return left_side, right_side


def find_best_split(data):
    '''
    Loops over the entire dataset's features,
    and calculate the information gain for each feature,
    then pick the highest one

    returns:
        - best split feature
        - question
    '''

    dataset_gini = calc_gini(data)
    dataset_size = data.shape[0]

    highest_information_gain = 0
    best_feature_idx, best_feature = None, None
    left_partition = None
    right_partition = None

    for row in data.itertuples(index=False):  # loop over entire dataset
        for feature_idx, feature in enumerate(list(row)[:-1]):  # loop over all features except the last one (class)

            # partition the dataset using that feature
            left_side, right_side = partition(data, feature_idx, feature)
            left_side_size = left_side.shape[0]
            right_side_size = right_side.shape[0]

            # calculate gini for left + right side
            try:
                left_side_gini = calc_gini(left_side)
                right_side_gini = calc_gini(right_side)
            except ZeroDivisionError:
                continue

            # calculate information gain,
            # update if higher than current (with question too)
            IG = dataset_gini - (((left_side_size/dataset_size) * left_side_gini) + ((right_side_size/dataset_size) * right_side_gini))
            if IG > highest_information_gain:
                highest_information_gain = IG
                left_partition = left_side
                right_partition = right_side
                best_feature_idx, best_feature = feature_idx, feature

    return best_feature_idx, best_feature, left_partition, right_partition


# binary tree stuff

class Node:
    def __init__(self, question, left, right):
        self.question = question
        self.left = left
        self.right = right


def build_tree(data, max_depth, curr_depth=0):

    if max_depth + 1 == curr_depth:
        return

    best_feature_idx, best_feature, left_partition, right_partition = find_best_split(data)
    left_side = build_tree(left_partition, max_depth, curr_depth+1)
    right_side = build_tree(right_partition, max_depth, curr_depth+1)

    return Node(question=[best_feature_idx, best_feature], left=left_side, right=right_side)


def tree_printer(root_node, spacing=''):

    if (root_node.left is None) and (root_node.right is None):
        return

    print(spacing + 'Is the feature #{0} from the dataset <= {1}'.format(root_node.question[0], root_node.question[1]))

    tree_printer(root_node.left, spacing + '  ')
    tree_printer(root_node.right, spacing + '  ')


def predict(root, input, d):
    if (root.left is None) and (root.right is None):

        class_0_count = d[d['class'] == 0].shape[0]
        class_1_count = d[d['class'] == 1].shape[0]

        if class_0_count > class_1_count:
            return 0
        else:
            return 1

    feature_idx = root.question[0]
    feature = root.question[1]

    left_partition, right_partition = partition(d, feature_idx, feature)

    if input[feature_idx] <= feature:
        return predict(root.left, input, left_partition)
    else:
        return predict(root.right, input, right_partition)


tree_root = build_tree(data, max_depth=3)

tree_printer(tree_root)

array = data.values
X = array[:, 0:8]
Y = array[:, 8]
num_correct = 0
for x, y in zip(X, Y):
    if predict(tree_root, x, data) == int(y):
        num_correct += 1

accuracy_own_implementation = (num_correct/len(X)) * 100

# 77.60416666666666
print(accuracy_own_implementation)