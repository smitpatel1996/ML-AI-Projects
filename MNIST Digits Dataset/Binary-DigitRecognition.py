import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

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
Y_train = (mnist_train.iloc[:,0]).to_frame()
Y_test = (mnist_test.iloc[:,0]).to_frame()

# Data Exploration
# plot = Plot()
# plot.perform(X_train)
# print(Y_train.iloc[0,:].to_numpy())

