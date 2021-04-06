
Machine Learning from Scratch
-------------

I have implemented common ML algorithms with Numpy, and verified the correctness of my implementations using sk-learn.

Inside each algorithm's code, you'll always find this 4-step experiment:   
1. load the relevant dataset
2. solve the problem using sk-learn and 1.
3. solve the problem using my own implementation and 1.
4. assert that 2. and 3. are equal, by:  
    - comparing the **accuracy on the entire dataset** in a **classification** scenario. 
    - taking **one sample from the dataset** randomly and comparing the predictions in a **regression** scenario.
    
    
Datasets
-------------

* For **binary classification**, I used the [Pima Indians Diabetes](https://www.kaggle.com/uciml/pima-indians-diabetes-database).
* For **regression**, I used the [Boston Housing Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#boston-dataset).