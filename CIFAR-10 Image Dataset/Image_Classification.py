from PIL import Image
from PIL import ImageEnhance
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

cifar = keras.datasets.cifar10
(X_train_full, Y_train_full), (X_test, Y_test) = cifar.load_data()

img = Image.fromarray(X_train_full[50].astype('uint8'), 'RGB').convert('L')
enhance = ImageEnhance.Sharpness(img)
img = enhance.enhance(2.0)
img.save('greyscale.png')
gray_np = np.array(img)
print(gray_np)
print(Y_train_full[50])