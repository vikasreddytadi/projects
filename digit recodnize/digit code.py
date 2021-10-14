import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()
print(len(X_train))
print(len(X_test))
X_train[0].shape

#by divide 255 all lie btw 0 to 1 to get accurate
X_train = X_train / 255
X_test = X_test / 255


#to convert 2 dim to 1 dim
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)


model.evaluate(X_test_flattened, y_test)
