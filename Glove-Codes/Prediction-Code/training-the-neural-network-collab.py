#training -one model-normalized data

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow
# import tflite_runtime.interpreter as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential,load_model
from keras.layers import  Flatten, Dense, Conv1D, MaxPool1D, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn import metrics

# from tensorflow.python.keras.callbacks import TensorBoard

data=pd.read_csv(r'/content/nadeen_dataset_8-7-23-normalizedMODIFIED.csv')

train=data[(data['person_id']<=5)]
valid=data[(data['person_id']>5) & (data['person_id']<=7)]
test=data[(data['person_id']>7)]
               ###################Train##################
trainx=train[['F0_norm','F1_norm','F2_norm','F3_norm','F4_norm', 'norm_ax', 'norm_ay', 'norm_az', 'Index_Touch', 'Middle_Touch']]
trainy=train[['Gesture_id', 'Output']].drop_duplicates()['Output']
G_L_train=train[['Gesture_label']].drop_duplicates()

                  ########Converting to Matrix########
trainx=np.array(trainx)
trainy=np.array(trainy)
G_L_train=np.array(G_L_train)
              ####### convert trainy to one hot code ########
from keras.utils import np_utils
trainy=np_utils.to_categorical(trainy,29)

#######################################################################################

          ###################Test##################
testx=test[['F0_norm','F1_norm','F2_norm','F3_norm','F4_norm', 'norm_ax', 'norm_ay', 'norm_az', 'Index_Touch', 'Middle_Touch']]
testy=test[['Gesture_id','Output']].drop_duplicates()['Output']
G_L_test=test['Gesture_label'].drop_duplicates()
                  ########Converting to Matrix########
testx=np.array(testx)
testy=np.array(testy)
G_L_test=np.array(G_L_test)

            ####### convert testy to one hot code ########
from keras.utils import np_utils
testy=np_utils.to_categorical(testy,29)

#########################################################################################

                   ###################Validation##################
validx=valid[['F0_norm','F1_norm','F2_norm','F3_norm','F4_norm', 'norm_ax', 'norm_ay', 'norm_az', 'Index_Touch', 'Middle_Touch']]
validy=valid[['Gesture_id','Output']].drop_duplicates()['Output']
G_L_valid=valid[['Gesture_label']].drop_duplicates()
                  ########Converting to Matrix########
validx=np.array(validx)
validy=np.array(validy)
G_L_valid=np.array(G_L_valid)

            ####### convert validy to one hot code ########
from keras.utils import np_utils
validy=np_utils.to_categorical(validy,29)
##########################################################################################

                  ###################Reshaping Data##########################
print(f" Before Shape Train X : {trainx.shape}  ")
print(f" Before Shape Train y : {trainy.shape}  ")
print(f" Before Shape Test X  :  {testx.shape}  ")
print(f" Before Shape Test y  :  {testy.shape}  ")
print(f" Before Shape Valid X : {validx.shape}  ")
print(f" Before Shape Valid Y : {validy.shape}  ")
print(f" Before Shape train Gesture_label  :  {G_L_train.shape}  ")
print(f" Before Shape Test Gesture_label  :  {G_L_test.shape}  ")
print(f" Before Shape Valid Gesture_label  :  {G_L_valid.shape}  ")

trainx_shape=np.reshape(trainx,(174,100, 10))
trainy_shape=trainy.reshape(174,29)
testx_shape=np.reshape(testx,(58,100,10))
testy_shape=testy.reshape(58,29)
validx_shape=np.reshape(validx,(58,100,10))
validy_shape=validy.reshape(58,29)

############################################################################################

                ########Sequential model########
cnn_model=tensorflow.keras.models.Sequential()

                #############CNN Layers#########
cnn_model.add(Conv1D(filters=32,kernel_size=(5,),padding='same',activation=tensorflow.nn.relu,input_shape=trainx_shape.shape[1:]))
cnn_model.add(Conv1D(filters=64,kernel_size=(5,),padding='same',activation=tensorflow.nn.relu))
cnn_model.add(Conv1D(filters=128,kernel_size=(5,),padding='same',activation=tensorflow.nn.relu))

            #######CNN with MaxPooling layer#########
cnn_model.add(MaxPool1D(pool_size=(5,),strides=5,padding='same'))
cnn_model.add(Dropout(0.5))

               #############Flatten the Output#########
cnn_model.add(Flatten())
cnn_model.add(Dense(units=256,activation=tensorflow.nn.relu))
cnn_model.add(Dense(units=512,activation=tensorflow.nn.relu))

             #############Softmax as Last layer#########
cnn_model.add(Dense(units=29,activation='softmax'))

      ##########create file to set all analysis on it##########

cnn_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
cnn_model.summary()

#________________________________________________________________________

che=ModelCheckpoint(filepath='weights.hdf5',verbose=1,save_best_only=True)
hist=cnn_model.fit(trainx_shape, trainy_shape, epochs=200, batch_size = 128, validation_data = (validx_shape, validy_shape),callbacks=[che],shuffle=True)

#_______________________________________________________________________

# Evaluate model
#_______________________________________________________________________
cnn_model.load_weights('weights.hdf5')
train_acc=cnn_model.evaluate(trainx_shape,trainy_shape,verbose=0)
accuracy=100*train_acc[1]
loss=train_acc[0]
print(f' Accuracy_train = {round(accuracy,2)}%  Loss_train = {round(loss,2)} ')
val_acc=cnn_model.evaluate(validx_shape,validy_shape,verbose=0)
accuracy=100*val_acc[1]
loss=val_acc[0]
print(f' Accuracy_val = {round(accuracy,2)}%  Loss_val = {round(loss,2)} ')
test_acc=cnn_model.evaluate(testx_shape,testy_shape,verbose=0)
accuracy=100*test_acc[1]
loss=test_acc[0]
print(f' Accuracy_test = {round(accuracy,2)}%  Loss_test = {round(loss,2)} ')

cnn_model.save('CNN_MODLE.h5')

#prediction
#_______________________________________________________________________

pred=cnn_model.predict(testx_shape).round()
pred=np.argmax(pred,axis=1)
testy_shape=np.argmax(testy_shape,axis=1)
print(metrics.classification_report(testy_shape,pred))

# visualize accuracy of model
#_______________________________________________________________________

#  "Accuracy"
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
# "Loss"
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()