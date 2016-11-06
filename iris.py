# Adding libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Loading Iris Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# Properties of the Data
# print "Shape of the Dataset:",dataset.shape
# 150 rows and 5 Columns
# Output - Shape of the Dataset: (150, 5)

# Seeing the Data
# print dataset.head(10)
# Output
#    sepal-length  sepal-width  petal-length  petal-width        class
# 0           5.1          3.5           1.4          0.2  Iris-setosa
# 1           4.9          3.0           1.4          0.2  Iris-setosa
# 2           4.7          3.2           1.3          0.2  Iris-setosa
# 3           4.6          3.1           1.5          0.2  Iris-setosa
# 4           5.0          3.6           1.4          0.2  Iris-setosa
# 5           5.4          3.9           1.7          0.4  Iris-setosa
# 6           4.6          3.4           1.4          0.3  Iris-setosa
# 7           5.0          3.4           1.5          0.2  Iris-setosa
# 8           4.4          2.9           1.4          0.2  Iris-setosa
# 9           4.9          3.1           1.5          0.1  Iris-setosa

# Getting statistical details about the data:
# print dataset.describe()
# In the output, we get data pertaining to properties of the attributes.
#        sepal-length  sepal-width  petal-length  petal-width
# count    150.000000   150.000000    150.000000   150.000000
# mean       5.843333     3.054000      3.758667     1.198667
# std        0.828066     0.433594      1.764420     0.763161
# min        4.300000     2.000000      1.000000     0.100000
# 25%        5.100000     2.800000      1.600000     0.300000
# 50%        5.800000     3.000000      4.350000     1.300000
# 75%        6.400000     3.300000      5.100000     1.800000
# max        7.900000     4.400000      6.900000     2.500000

# print dataset.groupby('class').size()
# Output : We get the number of results per class.
# class
# Iris-setosa        50
# Iris-versicolor    50
# Iris-virginica     50

# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()

# dataset.hist()
# plt.show()

# scatter_matrix(dataset)
# plt.show()

# Evaluating Dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# 10-fold cross validation!
num_folds = 10
num_instances = len(X_train)
seed = 7
scoring = 'accuracy'

# Algorithm Comparision.
models = []
models.append(('LogisticRegression', LogisticRegression()))
models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
models.append(('KNeighborsClassifier', KNeighborsClassifier()))
models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
models.append(('GaussianNB', GaussianNB()))
models.append(('SVC', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
	cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Output :
# LogisticRegression: 0.966667 (0.040825)
# LinearDiscriminantAnalysis: 0.975000 (0.038188)
# KNeighborsClassifier: 0.983333 (0.033333) ###THE MOST EFFICIENT
# DecisionTreeClassifier: 0.975000 (0.038188)
# GaussianNB: 0.975000 (0.053359)
# SVC: 0.991667 (0.025000)

# Plotting Algorithms to compare
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

# Make predictions of the KNeighborsClassifier on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Output : We find that our final accuracy is 0.9 (90%)
# 0.9
# [[ 7  0  0]
#  [ 0 11  1]
#  [ 0  2  9]]
#                  precision    recall  f1-score   support
#
#     Iris-setosa       1.00      1.00      1.00         7
# Iris-versicolor       0.85      0.92      0.88        12
#  Iris-virginica       0.90      0.82      0.86        11
#
#     avg / total       0.90      0.90      0.90        30
