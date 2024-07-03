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
14.Clustering(unsupervised learning)
---Intercluster and intracluster
we want intercluster to be max and intracluster to be min
-----Partion based clustering(K means OR loyds algorithm)
-----hierarchial clustering
------ well seperated 
------centre based
------- density based (DBSCAN)
k means++ is used to choose initial 2 centroid in k means clustering

--hierarchial clustering --agglomeratve(down to up) and divisive(up to down)
15.Fine tuning of data models can be done with GridSearchCv(especially svm)
16.Often time we encounter imbalanced data sets
17.NLTK(natutal language tool kit)
Notebooks Used in the course/Heart Failure Prediction System/Heart Failure Prediction 📚.ipynb
https://github.com/ayush714/ML001-Project-Sources-Code-and-Learning-Materials/blob/3af2e069c065ce3d078d8d4c15aadedf4e3c9d8f/Notebooks%20Used%20in%20the%20course/Heart%20Failure%20Prediction%20System/Heart%20Failure%20Prediction%20%F0%9F%93%9A.ipynb
https://github.com/ayush714/ML001-Project-Sources-Code-and-Learning-Materials/blob/3af2e069c065ce3d078d8d4c15aadedf4e3c9d8f/Notebooks%20Used%20in%20the%20course/Spam%20Detector%20System/Spam%20Detector%20System.ipynb
machine learning materials:https://github.com/ayush714/ML001-Project-Sources-Code-and-Learning-Materials/tree/3af2e069c065ce3d078d8d4c15aadedf4e3c9d8f/ML001%20Lecture%20Notes
TensorFlow 
1.Neraul Network-a form of Ml which is a layered represenataion of data
different types of Ml-supervised,unsupervised,reinforcement
2.Two main components of tensor flow

---graphs,sessions
3.a tensor is a generalizaton of vectors and matrices to higher dimesnions
4.an epoch is simply one streeam of our entire dataset.The number of epochs is the number of times our model will see the entire dataset
5.Hidden Markov Models-works with probabilituies to predict furture  event or state(weather predictiomn)
this model works with  probability distribution
it needs -states,obseravtion probability,transition probability
activation functions in NN
---Relu,Tanh,Sigmoid

6.optimizer is the function that does the back propogation for us(maps the cost function and goes towards global minimum)
7.relu-rectifier linear unit
all the neural networks seen before were feed forward neural networks
8.NLP(Natural language processing)-RNN recurrent nueral networks
algorithms--1.bag of words2.word embeddings
word embedings-clssify words as vectors in a 3D plane,similar words lie close to each other in this 3d plane(the similar word vectors point in the similar directions)
LSTM(long short term memory) is a type of layer in the RNN
first text goes through embeddings layer and then througH LSTM
