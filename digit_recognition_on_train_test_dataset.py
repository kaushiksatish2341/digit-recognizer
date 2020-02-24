import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

a=pd.read_csv('train.csv')
b=pd.read_csv('test.csv')
print(a)
print(b)
dataseta=a.values
datasetb=b.values

Xa=dataseta[:,1:]
ya=dataseta[:,0]



X_scalea=scalar.fit_transform(Xa)


X_train,X_val,y_train ,y_val= train_test_split(X_scalea,ya,test_size=0.2)
#X_test, y_test = train_test_split(X_scaleb,yb)





model = Sequential([
    Dense(512, activation='relu', input_shape=(784,)),
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    

hist = model.fit(X_train,y_train,
          batch_size=1024, epochs=25,
          validation_data=(X_val, y_val))

print("Accuracy on training set:{:.3f}".format(model.evaluate(X_train,y_train)[1]))
print("Accuracy on Validation set:{:.3f}".format(model.evaluate(X_val,y_val)[1]))
#print("Accuracy on test set:{:.3f}".format(model.evaluate(X_test,y_test)[1]))


