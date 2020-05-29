import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow,keras.layers import Dropout
from tensorflow.keras.datasets import mnist

(X_train,Y_train),(X_test,Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0],28*28).astype("float32")
X_test = X_test.reshape(X_test.shape[0],28*28).astype("float32")

X_train = X_train/255
X_test = X_test/255
print(X_train[0][150:175])

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

model = Sequential()
model.add(Dense(784,input_dim=784,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10,activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
history = model.fit(X_train,Y_train,validation_split=0.2,epochs=10,batch_size=128,verbose=2)

loss, acc = model.evaluate(X_train,Y_train)
v_loss,v_acc = model.evaluate(X_test,Y_test)
print("acc of training data is {:.2f}".format(acc))
print("acc of validation data is {:.2f}".format(v_acc))
