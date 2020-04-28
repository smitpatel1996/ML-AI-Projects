import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score as cvs

class Plot():
	def perform(self, dataFrame, rowIndex=0):
		row = (dataFrame.iloc[rowIndex,:]).to_numpy()
		#Program Specific Implementation
		plt.imshow(row.reshape(28, 28), cmap = matplotlib.cm.binary, interpolation="nearest")
		plt.axis("off")
		plt.show()
		
class Shuffle():
	def perform(self, dataFrame):
		return dataFrame.reindex(np.random.permutation(dataFrame.index))

class StratifiedValidation():
	def __init__(self, split):
		self.split = split
	def perform(self, model, attrSet, labelSet):
		skfolds = StratifiedKFold(n_splits=self.split)
		labels = np.array([])
		preds = np.array([])
		for train_index, test_index in skfolds.split(attrSet, labelSet):
			model_clone = clone(model)
			attr_trainFolds = attrSet[train_index]
			label_trainFolds = labelSet[train_index]
			attr_testFold = attrSet[test_index]
			label_testFold = labelSet[test_index]
			model_clone.fit(attr_trainFolds , label_trainFolds)
			label_pred = model_clone.predict(attr_testFold)
			labels = np.append(labels, label_testFold)
			preds = np.append(preds, label_pred)
		return (labels, preds)        

### ==== ACTUAL IMPLEMENTATION ==== ###
mnist_train = pd.read_csv("mnist_train.csv", sep=",", names=range(1,786))
mnist_test = pd.read_csv("mnist_test.csv", sep=",", names=range(1,786))

# Random Shuffling of the data
shuffle = Shuffle()
mnist_train = shuffle.perform(mnist_train)
mnist_test = shuffle.perform(mnist_test)

# Splitting Dataset into Training and Testing Subsets
X_train = mnist_train.iloc[:,1:]
X_test = mnist_test.iloc[:,1:]
Y_train = mnist_train.iloc[:,0]
Y_test = mnist_test.iloc[:,0]

# Data Exploration
# plot = Plot()
# plot.perform(X_train)
# print(Y_train.iloc[0])

# 5 vs not-5 -- Binary Classification
Y_train_5 = (Y_train == 5)
Y_test_5 = (Y_test == 5)

sgd_clf_model = linear_model.SGDClassifier(random_state=50)

stratifiedValidation = StratifiedValidation(3)
labels, preds = stratifiedValidation.perform(sgd_clf_model, X_train.to_numpy(), Y_train_5.to_numpy())
print(confusion_matrix(labels, preds))