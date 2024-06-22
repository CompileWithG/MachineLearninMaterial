# MachineLearninMaterial
MachineLearning Notes
0.https://github.com/Spartan-119/Python-for-Data-Science.git
1.Features(feature vectors) are the input data we feed into the model to get the predicted output.
2.One-Hot Encoding (categorical values)
--Dummy variables are binary (0 or 1) variables created to represent a categorical variable with two or more categories. For each category, a dummy variable is created, where the variable is equal to 1 if the category is present, and 0 if it is not.
3.types of classification (binary and multiclass)
---01 Classification-predicts discrete classes 
---02 Regression-predicts continuous values
4.Dividing dataset into training ,validation and testing
from sklearn
5.A regression is a statistical technique that relates a dependent variable to one or more independent (explanatory) variables. A regression model is able to show whether changes observed in the dependent variable are associated with changes in one or more of the explanatory variables.
6.SVM(Support vector machine)--creates hyperplanes to separate data into classes .(kernel trick method)
from sklearn.svm import SVR
https://youtu.be/_YPScrckx28?si=QZZRrKHq4xRk-nvR
7.sigmoid function (between 0 and 1) used in logistic regression
--logistic regression
8.need for dimensionality reduction
when there are lot of dimesnions(features) WE decrease some dimensions to be able to handle the accuracy,load ,storage
PCA --principal componenent analysis
9.BIAS AND VARIANCE 
high variance-overfitting
high bias-underfitiing
ideal model-low variance and low bias
10.  DECISION TREES 
------------
ENTROPY-MEASURE OF RANDOMNESS TO CHOOSE THE NODE(USED TO CHOOSE ROOT NODE)
entropyH(D)=-p(y)*log(p(y))
range-0 to 1
if entripy is 0 it is most suitable for leaf node 
if entropy is 1 it needs further splitting
------information gain
------gini impurity
11.ensemble learning ---1.boosting 2.bagging(bootstrap aggregiation) 3.stacking
The idea behind ensemble learning is that by combining multiple models, each with its strengths and weaknesses, the ensemble can achieve better results than any single model alone. Ensemble learning can be applied to various machine learning tasks, including classification, regression, and clustering.
---------bagging helps in reducing variance(the base model are high variance and low bias )
-----Random Forest(bagging algorith)==decision tress+ bagging+feature bagging+row and column sampling
in random forest we take decision trees as base models
----OOB(out of bag points)
----extremely randomised trees
Random Forest chooses the optimum split while Extra Trees chooses it randomly. However, once the split points are selected, the two algorithms choose the best one between all the subset of features. Therefore, Extra Trees adds randomization but still has optimization.
https://www.baeldung.com/cs/random-forest-vs-extremely-randomized-trees
above is the difference between random forest and extra trees
12.boosting---gradient boosting,XGBoost,AdaBoost
boosting reduces bias
in boosting base model have low variance and high bias9we additively combine these models)

13.gradient boosted decision trees(GBDT)
