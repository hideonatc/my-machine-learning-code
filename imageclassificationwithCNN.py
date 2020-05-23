import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.datasets import cifar10
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.models import Sequential
from keras.utils import to_categorical

seed = 10
np.random.seed(seed)

(X_train,Y_train),(X_test,Y_test) = cifar10.load_data()
X_train = X_train.astype("float32")/255
X_test = X_test.astype("float32")/255

Y_test_bk = Y_test
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),padding="same",input_shape=X_train.shape[1:],activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(64,kernel_size=(3,3),padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10,activation="softmax"))
model.summary()
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
history = model.fit(X_train,Y_train,validation_split=0.2,epochs=9,batch_size=128,verbose=2)
model.save("cifar10.h5")

tb = pd.crosstab(Y_test_bk.astype(int).flatten(),Y_pred.astype(int),rownames=["label"],colnames=["predict"])
Y_pred = model.predict_classes(X_test)
Y_prbs = model.predict_proba(X_test)
Y_test = Y_test.flatten()
df = pd.DataFrame({"label":Y_test,"predict":Y_pred})
df = df[Y_test!=Y_pred]
print(df.head())