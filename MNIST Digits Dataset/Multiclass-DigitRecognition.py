import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn import ensemble
from sklearn import multiclass
from sklearn import metrics
from sklearn.metrics import roc_curve as ROC_curve
from sklearn.metrics import precision_recall_curve as PR_curve
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score as cvs
from sklearn.model_selection import cross_val_predict as cvp

class Shuffle():
    def perform(self, dataFrame):
        return dataFrame.reindex(np.random.permutation(dataFrame.index))

class Plot():
    def perform(self, dataFrame, rowIndex=0):
        row = (dataFrame.iloc[rowIndex,:]).to_numpy()
        #Program Specific Implementation
        plt.imshow(row.reshape(28, 28), cmap = matplotlib.cm.binary, interpolation="nearest")
        plt.axis("off")
        plt.show()

class ScaleFeature():
    def perform(self, dataFrame, training=True):
        if(training):
            self.transformer = StandardScaler()
            self.transformer.fit(dataFrame)   
        npOutput = self.transformer.transform(dataFrame)
        return npOutput

class PreProcess():
    def __init__(self):
        self.scaleFeature = ScaleFeature()

    def perform(self, dataFrame, training=True):
        npOutput = self.scaleFeature.perform(dataFrame, training)
        return npOutput

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

class ExtractMetrics():
    def perform(self, labels, preds):
        print("Confusion Matrix: ")
        print(metrics.confusion_matrix(labels, preds))
        print("Classification report: ")
        print(metrics.classification_report(labels, preds))

class AnalyzeCurves():
    def __init__(self, split):
        self.split = split
    
    def __PvR(self, labelSet, scores, classes):
        aucSum = 0
        for i in classes:
            precisions, recalls, thresholds = PR_curve(labelSet[:, i], scores[:, i])
            plt.plot(recalls, precisions, label="Class: {}".format(i))
            aucSum = aucSum + metrics.auc(recalls, precisions)
        print("Average AUC for Precision Vs Recall: " + str(aucSum/len(classes)))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.axis([0, 1, 0, 1])
        plt.legend(loc="best")
        plt.title("Precision vs Recall")
        plt.grid(b=True, which='major', color='#666666', linestyle='--')
        plt.show()
    
    def __ROC(self, labelSet, scores, classes):
        aucSum = 0
        for i in classes:
            fpr, tpr, thresholds = ROC_curve(labelSet[:, i], scores[:, i])
            plt.plot(fpr, tpr, label="Class: {}".format(i))
            aucSum = aucSum + metrics.auc(fpr, tpr)
        print("Average AUC for ROC: " + str(aucSum/len(classes)))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.legend(loc="best")
        plt.title("ROC Curve")
        plt.grid(b=True, which='major', color='#666666', linestyle='--')
        plt.show()
            
    def perform(self, model, attrSet, labelSet, curve='PvR', method='decision_function'):
        classes = list(set(labelSet))
        n_classes = len(classes)
        labelSet = label_binarize(labelSet, classes=[*range(n_classes)])
        clf = multiclass.OneVsRestClassifier(model)
        scores = cvp(clf, attrSet, labelSet, cv=self.split, method=method)
        if(curve == 'PvR'):
            self.__PvR(labelSet, scores, classes)
        elif(curve == 'ROC'):
            self.__ROC(labelSet, scores, classes)
          
class PRAnalysis():
    def __init__(self, split):
        self.split = split
        
    def perform(self, model, attrSet, labelSet, method='decision_function'):
        classes = list(set(labelSet))
        n_classes = len(classes)
        labelSet = label_binarize(labelSet, classes=[*range(n_classes)])
        clf = multiclass.OneVsRestClassifier(model)
        scores = cvp(clf, attrSet, labelSet, cv=self.split, method=method)
        for i in classes:
            precisions, recalls, thresholds = PR_curve(labelSet[:, i], scores[:, i])
            plt.plot(thresholds, precisions[:-1], label="Precision: {}".format(i))
            plt.plot(thresholds, recalls[:-1], label="Recall: {}".format(i))
        plt.xlabel("Threshold")
        plt.ylim([0, 1])
        plt.legend(loc="best")
        plt.title("Precision & Recall vs Threshold")
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
trainDF = mnist_train.iloc[:,1:]
testDF = mnist_test.iloc[:,1:]
Y_train = mnist_train.iloc[:,0]
Y_test = mnist_test.iloc[:,0]

# Data Exploration
# plot = Plot()
# plot.perform(X_train)
# print(Y_train.iloc[0])

#PreProcessing the Subsets
preProcess = PreProcess()
X_train = preProcess.perform(trainDF)
X_test = preProcess.perform(testDF, False)
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()

#Initializing Models
#sgd_clf_model = linear_model.SGDClassifier(random_state=50, max_iter=10)
rf_clf_model = ensemble.RandomForestClassifier(random_state=50, n_estimators=10)

#Cross-Validating the Models.
validateModels = ValidateModels(3)
#sgd_preds = validateModels.perform(sgd_clf_model, X_train, Y_train, 'predictions')
rf_preds = validateModels.perform(rf_clf_model, X_train, Y_train, 'predictions')
extractMetrics = ExtractMetrics()
#extractMetrics.perform(Y_train, sgd_preds)
extractMetrics.perform(Y_train, rf_preds)

#Curves Analysis
analyzeCurves = AnalyzeCurves(3)
#analyzeCurves.perform(sgd_clf_model, X_train, Y_train)
analyzeCurves.perform(rf_clf_model, X_train, Y_train, 'ROC', 'predict_proba')