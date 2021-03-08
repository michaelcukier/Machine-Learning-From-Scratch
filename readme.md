
Machine Learning from Scratch
-------------

I have implemented common ML algorithms with Numpy, and verified the correctness of my implementations using sk-learn.

Inside each algorithm's code, you'll always find the same 4-step experiment:   
1. load the relevant dataset
2. solve the problem using sk-learn and 1.
3. solve the problem using my own implementation and 1.
4. assert that 2. and 3. results are equal

Datasets
-------------

* For **binary classification**, I used the [Pima Indians Diabetes](https://www.kaggle.com/uciml/pima-indians-diabetes-database).
* For **regression**, I used the [Boston Housing Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#boston-dataset).
* For **multi-class classification**, I used [x](x).


Algorithms
-------------


#### Linear Algorithms

* Classification

- [x] Logistic Regression (trained using SGD): [code](./logistic_regression.py)

* Regression

- [x] Linear Regression (trained using SGD): [code](./linear_regression.py)
- [ ] Linear Discriminant Analysis

#### Non-Linear Algorithms

- [ ] Classification and Regression Trees
- [ ] Naive Bayes
- [ ] Gaussian Naive Bayes
- [ ] k-Neareast Neightbors
- [ ] Learning Vector Quantization
- [ ] SVM

#### Ensemble Algorithms

- [ ] Boosting