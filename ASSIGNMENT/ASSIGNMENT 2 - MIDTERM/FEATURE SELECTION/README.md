# FEATURE SELECTION METHODS THAT WE HAVE USED :

<b>Once we are done with EDA and feature engineering , we need to now select which all features are important to be considered for modeling.
Here are few methods that we have tried to come out with features that will provide with the best model output.<b>
<ol>
  <li><b> *Recursive Feature Elimination :* <b><br>
First , the model takes few features into iteration and in that iteration , it provides weights to each feature and in the next iteration it takes some other features and compare the weight of the first iteration with the next iteration and eliminate the variables based on these weights. The one with maximum weights will be given as output depending on the number of features we want.  
<br>
In this , I am fitting the X and Y in the model(which is linearRegresion()) . 
<b>X<b> has all my variables except the target that is Appliances.
<b>Y<b> has the target.
I am asking it to output 22 features. 
<b>Output<b> - the 22 features with its rank which is based on how much it has impacted the appliances.
<br><li><b>Feature Importance Method :<b><br>
This is completely randomized tree which splits the X and the Y randomly .
n_estimators = number of trees in the model.
I am fitting the variables in the model and they provide me the important features which are based on weights.
<br><li><b>SelectKbest :<b><br>
In this model , we fit the input and the target in the model which calculates the p_value for us. As this method doesn't tell us which feature values more , this is not cosidered.<br>
We are using f_regression here. There are other estimators that can be used such as chi2 , f_classif , selectpercentile , selectfdr , selectfpr,selectfwe etc. but only f_regression suited our dataset. chi2 can also be used if there were no negative values. As T6 has negative values in it , chi2 method can't be used.
<br><li><b>SelectFromModel :<b><br>
This method is based on Lasso regularization.<br>
Most of the time their coefficients will be zero due to which they are used in case of dimensionality reduction.  Alpha is the deciding factor here. More the alpha value , less is the feature selected.
Mainly used on linear models.
<br><li><b>Low Variance(Based on score) :<b><br>
Here it calculates the variance of each column and compares it with the variance that is provided in the input.<br> 
Any variance less that the variance of the given input is removed.<br>
The biggest disadvantage about this technique is , it considers only one data frame. It either takes X or Y . It doesn't take both. So this method is mainly used for unsupervised learning and not for our dataset. 
<br><li><b>Forward Selection :<b><br>
The simplest data-driven model building approach is called forward selection. In this
approach, one adds variables to the model one at a time. At each step, each
variable that is not already in the model is tested for inclusion in the model. The
most significant of these variables is added to the model.Thus we begin with a model including the variable that is most significant
in the initial analysis, and continue adding variables until none of remaining
variables are significant. 
<br><li><b>Backward Selection : <b><br>
Forward selection has drawbacks, including the fact that each addition of a new
variable may render one or more of the already included variables non-significant. An
alternate approach which avoids this is backward selection. Under this approach, one
starts with fitting a model with all the variables of interest . Then the least significant variable is dropped, so long as it is not significant at
our chosen critical level. We continue by successively re-fitting reduced models and
applying the same rule until all remaining variables are statistically significant.
<br><li><b> Stepwise Regression :<b><br>
Performing both the above models together.
It combines both Forward and Backward eliminations and drops/adds variables
based on their statistical.</li>
<ol>
