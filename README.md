# Machine learning bootcamp 🤖

![Machine learning](https://i0.wp.com/datascientest.com/wp-content/uploads/2020/11/Machine-learning-def-.png?fit=1025%2C563&ssl=1 "Machine learning")

This bootcamp has been designed by the [42 AI](http://42ai.fr) association and is a great first step into machine learning throught **supervised leaning**.

The purpose of it is to learn **linear regression**, **logistic regression** and a lot of very interesting notions (loss, regularization, gradient descent, ...) by coding programs that can learn and perform predictions

The full content of the bootcamp is available on 42 AI's GitHub page : https://github.com/42-AI/bootcamp_machine-learning

# Detailed content of the modules

## Module 00 : Introduction to linear regression

This module is an introduction to **linear regression**. The main notions that are learned in this module are :

* **Matrix and vector operations** : scalar product, addition , substractions, multiplications, divisions
* **Basic statistics** : mean, median, percentiles, quartiles, variance, standard deviation
* **Prediction** : How to use a linear function and basic vector operation to predict values from features
* **Loss function** : calculate how bad is our prediction compared to the real data. Multiple loss functions (MSE, RMSE, MAE, R2 score)
* **Plotting** : Using matplotlib to draw our first data visualizations

## Module 01 : Gradient descent, normalization

This module is about completing the first one by going further into linear regression. In this module we learn :

* **Gradient** : what is a gradient vector, how it is calculated and how it can be used to adjust the weights of the prediction function
* **Gradient descent** : Optimization algorithm based on gradient calculation
* **Normalization** : Why normalize data and how to use z-score and minmax normalization
* **Wrap up all functions into a python class and use it !** : Start using all we learned to solve small machine learning problems (with univariate linear regression)

## Module 02 : Multivariate linear regression

This module is about multivariate linear regression. We will tackle the problem of having multiple features and using them to do a better prediction. The main notions are :

* **Prediction** : Adapt it with a linear function that uses multiple features and weights
* **Loss function** : Understand the impact on the loss function formula that we used before (no impact)
* **Gradient descent** : Adapt gradient descent to properly compute gradient vector and adapt all weights
* **Wrap up all functions into a python class and use it !** : Test our updated classes with multiple features dataset
* **Polynomial models** : Introduction to polynomial models, what it is, why, when and how to use them.
* **Overfitting** : Introduction to the overfitting problem

## Module 03 : Logistic regression

This module introduces classification with logistic regression. We will tackle the problem of having multiple features and using them to do a better prediction. The main notions are :

* **Sigmoid function** : What it is and why use it in classification problems for prediction
* **Cross entropy loss** : Applying a new loss function to estimate the prediction in a logistic regression model
* **Wrap up all functions into a python class and use it !** : Test our updated classes with a simple classifiation problem
* **One versus all** : Understand how to use a logistic regression to classify more than 2 classes
* **Model metrics and confusion matrix** : Understand accuracy, precision, recall and f1 score to evaluate our model.

## Module 04 : Regularization

This final module is about regularization. We can use regularization to handle the problem of overfitting. This module simply review all previous work on linear and logistic regression and add the possibility to apply a penalty on the loss function to **regularize** the final weights.

* **Adapt linear and logistic regression** : Adapt loss functions and gradient descent to take a **L2** penalty in account
* **Use regularization in 'real' conditions** : the final exercises are about using Ridge regression and regularized logistic regression with specific datasets and to observe the effect of it.

## Quick notes

This bootcamp is very rich and I really enjoyed learning by doing it. My code is not very clean on the first modules, and I realize that it can be improved a lot even on the final exercises, especially regarding data preparation, data augmentation (sometimes I added unecessary features and I shouldn't have), multiprocessing (the idea is ok but I don't perform it very well) and probably many other things.

I will try to improve my code on the next projects !

# Libraries used

* Numpy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn (to check my functions against it)

# Resources

* [42AI's GitHub](https://github.com/42-AI/bootcamp_machine-learning) : The bootcamp itself is very documented
* [Coursera](https://www.coursera.org/learn/machine-learning) : Andrew Ng's supervised learning course
