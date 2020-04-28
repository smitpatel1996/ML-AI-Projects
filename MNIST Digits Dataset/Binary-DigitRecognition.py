import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn import ensemble
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score as cvs
from sklearn.model_selection import cross_val_predict as cvp
from sklearn.metrics import precision_score as ps, recall_score as rs, f1_score as f1s
from sklearn.metrics import precision_recall_curve as PR_curve
from sklearn.metrics import roc_curve as ROC_curve
from sklearn.metrics import roc_auc_score as ROC_auc

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

class CustomStratifiedValidation():
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

class ValidateModels():
    def __init__(self, split):
        self.split = split
        
    def perform(self, model, attrSet, labelSet, returnType='scores'):
        if(returnType == 'scores'):
            scores = cvs(model, attrSet, labelSet, scoring="accuracy", cv=self.split)
            print("Accuracy Estimate: MEAN +/- STD = " + str(np.round(scores.mean(),3)) + " +/- " + str(np.round(scores.std(),3)))
        elif(returnType == 'predictions'):
            preds = cvp(model, attrSet, labelSet, cv=self.split)
            return preds
        elif(returnType == 'instanceScore'):
            scores = cvp(model, attrSet, labelSet, cv=self.split, method="decision_function")
            return scores

class ExtractMetrics():
    def perform(self, labels, preds):
        print('Confusion Matrix: ')
        print(confusion_matrix(labels, preds))
        print("Precision: " + str(ps(labels, preds)))
        print("Recall: " + str(rs(labels, preds)))
        print("F1_Score (Harmonic Mean of PS & RS): " + str(f1s(labels, preds)))

class ROCAnalysis():
    def __init__(self):
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        
    def perform(self, modelName, labels, scores):
        print("Area Under the ROC_Curve for " + modelName + ": " + str(ROC_auc(labels, scores)))
        fpr, tpr, thresholds = ROC_curve(labels, scores)
        plt.plot(fpr, tpr, label = modelName)
        plt.legend(loc="lower right")
  
class PRAnalysis():
    def perform(self, labels, scores, plot='PRvT'):
        precisions, recalls, thresholds = PR_curve(labels, scores)
        if(plot == 'PRvT'):
            plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
            plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
            plt.xlabel("Threshold")
            plt.legend(loc="upper left")
            plt.ylim([0, 1])
        elif(plot == 'PvR'):
            precisions, recalls, thresholds = PR_curve(labels, scores)
            plt.plot(recalls, precisions, "b-")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.axis([0, 1, 0, 1])
        plt.grid(b=True, which='major', color='#666666', linestyle='--')
        plt.show()

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

X_train = X_train.to_numpy()
Y_train_5 = Y_train_5.to_numpy()

#Initializing Models
sgd_clf_model = linear_model.SGDClassifier(random_state=50)
rf_clf_model = ensemble.RandomForestClassifier(random_state=50, n_estimators=10)

#Cross-Validating the Models.
validateModels = ValidateModels(3)
extractMetrics = ExtractMetrics()
sgd_preds = validateModels.perform(sgd_clf_model, X_train, Y_train_5, 'predictions')
extractMetrics.perform(Y_train_5, sgd_preds)
rf_preds = validateModels.perform(rf_clf_model, X_train, Y_train_5, 'predictions')
extractMetrics.perform(Y_train_5, rf_preds)
#Performing ROC Analysis to further validate model performances.
sgd_scores = validateModels.perform(sgd_clf_model, X_train, Y_train_5, 'instanceScore')
rf_scores = (cvp(rf_clf_model, X_train, Y_train_5, cv=3, method="predict_proba"))[:,1]
rocAnalysis = ROCAnalysis()
rocAnalysis.perform("SGD", Y_train_5, sgd_scores)
rocAnalysis.perform("Random Forest", Y_train_5, rf_scores)
plt.show()
#Shows that Random Forest is the Best Model.

#Adjusting the PvR tradeoff as per the problem statement using PRvT plot.
prAnalysis = PRAnalysis()
prAnalysis.perform(Y_train_5, rf_scores, 'PvR')