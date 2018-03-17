# TYPES OF MODELS
## REGRESSION MODELS

### Simple Linear Regression

Simple linear regression is a statistical method that allows us to summarize and study relationships between two continuous (quantitative) variables:
One variable, denoted x, is regarded as the predictor, explanatory, or independent variable.
The other variable, denoted y, is regarded as the response, outcome, or dependent variable.
Because the other terms are used less frequently today, we'll use the "predictor" and "response" terms to refer to the variables encountered in this course. The other terms are mentioned only to make you aware of them should you encounter them in other arenas. Simple linear regression gets its adjective "simple," because it concerns the study of only one predictor variable. In contrast, multiple linear regression, which we study later in this course, gets its adjective "multiple," because it concerns the study of two or more predictor variables.

### Mutiple Linear Regression

Multiple linear regression attempts to model the relationship between two or more explanatory variables and a response variable by fitting a linear equation to observed data. 
Every value of the independent variable x is associated with a value of the dependent variable y. The population regression line for p explanatory variables x1, x2, ... , xp is defined to be  y = 0 + 1x1 + 2x2 + ... +  pxp. This line describes how the mean response y changes with the explanatory variables. The observed values for y vary about their means y and are assumed to have the same standard deviation . The fitted values b0, b1, ..., bp estimate the parameters 0, 1, ..., p of the population regression line.
Since the observed values for y vary about their means y, the multiple regression model includes a term for this variation. In words, the model is expressed as DATA = FIT + RESIDUAL, where the "FIT" term represents the expression 0 +  1x1 +  2x2 + ... pxp. 
The "RESIDUAL" term represents the deviations of the observed values y from their means y, which are normally distributed with mean 0 and variance . The notation for the model deviations is .

### Random Forest Regressor

A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset 
and use averaging to improve the predictive accuracy and control over-fitting. 
The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default).

### Nueral Network

Multi Layer Perceptron (MLP) :
This is a supervised learning algorithm in which we give n dimension input and n dimension output. The model will train itself accordingly, depending on the weights that is given to each input and give us the proper output. The number of hidden layers between the input and the output can be tuned by us.
In this project , MLP regressor with 'sgd' solver is applied . 
Input is stored in X which contains all features except for Appliances that are scaled using MinMaxscaler  . 
Target which is Appliances is stored in Y  after scaling. 
Test and Train is divided into 30 ,70 ratio respectively.
Since the optimizer used is 'sgd' we need to provide the learning rate. 
The hidden_layer_sizes , which is the number of neurons which doesn't include the input and output sizes.


### Gradient Boosting Machine
Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.
Gradient boosting involves three elements:
A loss function to be optimized.
A weak learner to make predictions.
An additive model to add weak learners to minimize the loss function.


## Classification MODELS

### Support Vector Classifier

In machine learning, support vector machines (SVMs, also support vector networks are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. 
Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear 
classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting). 

## Learning objectives and outcomes
Upon completion of this lesson, you should be able to do the following:

Know how to calculate a confidence interval for a single slope parameter.
Be able to interpret the coefficients of a model.
Understand what the scope of the model is.
Understand the calculation and interpretation of R2.
Understand the calculation and use of adjusted R2. 
Understand the calculation of RMSE, MAE, MAPE
