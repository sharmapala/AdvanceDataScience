# Model validation using Hyperparameter and Cross validation on Random Forest (Final selected model)

### Hyper parameter
Basic recipe for applying a supervised machine learning model:

1. Choose a class of model
2. Choose model hyperparameters
3. Fit the model to the training data
4. Use the model to predict labels for new data <br>

Model's performance can be found using what's known as a holdout set: that is, we hold back some subset of the data from the training of the model, and then use this holdout set to check the model performance. 
This splitting can be done using the train_test_split utility in scikit-Learn.

### Model validation via cross-validation
One disadvantage of using a holdout set for model validation is that we have lost a portion of our data to the model training. This is not optimal, and can cause problems – especially if the initial set of training data is small. <br>
One way to address this is to use cross-validation; that is, to do a sequence of fits where each subset of the data is used both as a training set and as a validation set.  <br>
Here we do two validation trials, alternately using each half of the data as a holdout set.Scikit-Learn implements a number of useful cross-validation schemes that are useful in particular situations; these are implemented via iterators in the <b> cross_validation module</b>. For example, we might wish to go to the extreme case in which our number of folds is equal to the number of data points: that is, we train on all points but one in each trial. 
This type of cross-validation is known as leave-one-out cross validation.

### Cross Validation Techniques
Cross Validation is a very useful technique for assessing the effectiveness of your model, particularly in cases where you need to mitigate overfitting. It is also of use in determining the hyper parameters of your model, in the sense that which parameters will result in lowest test error. <br>
An error estimation for the model is made after training, better known as evaluation of residuals.
In this process, a numerical estimate of the difference in predicted and original responses is done, also called the training error. However, this only gives us an idea about how well our model does on data used to train it. o, the problem with this evaluation technique is that it does not give an indication of how well the learner will generalize to an independent/ unseen data set. Getting this idea about our model is known as Cross Validation.
<br>

### 1.Holdout Method
The error estimation then tells how our model is doing on unseen data or the validation set. 
This is a simple kind of cross validation technique, also known as the holdout method. 
It still suffers from issues of high variance. This is because it is not certain which data points will end up in the validation set and 
the result might be entirely different for different sets.
 
### 2.K-Fold Cross Validation
 By reducing the training data, we risk losing important patterns/ trends in data set, 
 which in turn increases error induced by bias. In K Fold cross validation, the data is divided into k subsets.The holdout method is 
 repeated k times, such that each time, one of the k subsets is used as the test set/ validation set and the other k-1 subsets are put 
 together to form a training set. This significantly reduces bias as we are using most of the data for fitting, and also significantly
 reduces variance as most of the data is also being used in validation set. <b>As a general rule and empirical evidence, K = 5 or 10 is
 generally preferred</b>
 
 ### 3.Stratified K-Fold Cross Validation
A slight variation in the K Fold cross validation technique is made, such that each fold contains approximately the same 
percentage of samples of each target class as the complete set, or in case of prediction problems, the mean response value is 
approximately equal in all the folds. This variation is also known as Stratified K Fold.

### 4.Leave-P-Out Cross Validation
This approach leaves p data points out of training data, i.e. if there are n data points in the original sample then, n-p samples are used to train the model and p points are used as the validation set. This is repeated for all combinations in which original sample can be separated this way, and then the error is averaged for all trials, to give overall effectiveness. 
A particular case of this method is when p = 1. This is known as Leave one out cross validation.

 ### Pipeline
 Pipeline of transforms with a final estimator.
 The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters. 
 For this, it enables setting parameters of the various steps using their names and the parameter name separated by a ‘__’, as in 
 the example below. A step’s estimator may be replaced entirely by setting the parameter with its name to another estimator,
 or a transformer removed by setting to None.
 
 ### GridSearchCV
 Exhaustive search over specified parameter values for an estimator.
 
 ### Bias - Variance trade off 
 One of the major aspects of training your machine learning model is avoiding overfitting. 
 The model will have a low accuracy if it is overfitting. This happens because your model is trying too hard to capture the 
 noise in your training dataset. By noise we mean the data points that don’t really represent the true properties of your data, 
 but random chance. The concept of balancing bias and variance, is helpful in understanding the phenomenon of overfitting.
 
For high-bias models, the performance of the model on the validation set is similar to the performance on the training set.<br>
For high-variance models, the performance of the model on the validation set is far worse than the performance on the training set.<br>

<ul>
  <li>The training score is everywhere higher than the validation score. This is generally the case: the model will be a better fit to data it has seen than to data it has not seen.</li>
  <li>For very low model complexity (a high-bias model), the training data is under-fit, which means that the model is a poor predictor both for the training data and for any previously unseen data.</li>
  <li>For very high model complexity (a high-variance model), the training data is over-fit, which means that the model predicts the training data very well, but fails for any previously unseen data.</li>
  <li>For some intermediate value, the validation curve has a maximum. This level of complexity indicates a suitable trade-off between bias and variance.</li>
</ul>
 
 ### Regularization
This is a form of regression, that constrains/ regularizes or shrinks the coefficient estimates towards zero. 
In other words, this technique discourages learning a more complex or flexible model, so as to avoid the risk of overfitting. One of the ways of avoiding overfitting is using cross validation, that helps in estimating the error over test set, 
and in deciding what parameters work best for your model.
 
