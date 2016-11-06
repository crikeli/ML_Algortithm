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
print "Shape of the Dataset:",dataset.shape
# 150 rows and 5 Columns
# Output - Shape of the Dataset: (150, 5)

# Seeing the Data
print dataset.head(10)
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
print dataset.describe()
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

print dataset.groupby('class').size()
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
