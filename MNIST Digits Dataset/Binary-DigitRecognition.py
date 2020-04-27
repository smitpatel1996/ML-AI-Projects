import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

mnist_train = pd.read_csv("mnist_train.csv", sep=",")
mnist_test = pd.read_csv("mnist_test.csv", sep=",")
X_train = mnist_train.iloc[:,1:]
X_test = mnist_test.iloc[:,1:]
Y_train = mnist_train.iloc[:,0]
Y_test = mnist_test.iloc[:,0]