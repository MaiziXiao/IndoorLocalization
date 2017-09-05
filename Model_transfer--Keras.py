import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras import regularizers
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#dataset parameter
BuildingID = 0
FloorID_1 = 0 #data to pretrain
FloorID_2 = 1 #data to be fine tuned
percentage = 0.3 #percentage of the second floor training data (when it's small, simulate not enough data situation)

#read training data1
dataset = pd.read_csv("trainingData.csv",header = 0)
dataset1 =  dataset[(dataset['FLOOR'] == FloorID_1)  & (dataset['BUILDINGID'] == BuildingID) ] #select the floor and building  & (dataset['BUILDINGID'] == BuildingID)
rss_values1 = np.asarray(dataset1.ix[:,0:-9]) #extract the features(delete the location, floor information etc.)
rss_values1[rss_values1 == 100] = -110 #make undefined measurements -110dbm
locations1 = np.asarray(dataset1.ix[:,-9:-7])
origin1 =np.amin(locations1,axis=0) #calculate the origin
room_size1 = np.amax(locations1, axis=0) - origin1 #size of the room
train_val_Y_1 = locations1 - origin1 #position respect to origin
train_val_X_1 = np.asarray(rss_values1, dtype=np.float64) #convert to numpy array

#read training data2
dataset2 =  dataset[(dataset['FLOOR'] == FloorID_2)  & (dataset['BUILDINGID'] == BuildingID) ] #select the floor and building  & (dataset['BUILDINGID'] == BuildingID)
rss_values2 = np.asarray(dataset2.ix[:,0:-9]) #extract the features(delete the location, floor information etc.)
rss_values2[rss_values2 == 100] = -110 #make undefined measurements -110dbm
locations2 = np.asarray(dataset2.ix[:,-9:-7])
origin2 =np.amin(locations2,axis=0) #calculate the origin
room_size2 = np.amax(locations2, axis=0) - origin2 #size of the room
train_val_Y_2 = locations2 - origin2 #position respect to origin
train_val_X_2 = np.asarray(rss_values2, dtype=np.float64) #convert to numpy array

#reading test data
test_dataset = pd.read_csv("validationData.csv",header = 0)
test_dataset =  test_dataset[(test_dataset['FLOOR'] == FloorID_2) & (test_dataset['BUILDINGID'] == BuildingID) ]  #& (test_dataset['BUILDINGID'] == BuildingID)
rss_values_test = np.asarray(test_dataset.ix[:,0:-9])
rss_values_test[rss_values_test == 100] = -110
test_locations= np.asarray(test_dataset.ix[:,-9:-7])
test_Y = test_locations - origin2
test_X = np.asarray(rss_values_test, dtype=np.float64)

#intermediate terms
num_data1 = len(train_val_X_1) #number of training points
num_data2 = len(train_val_X_2)
num_test = len(test_X) #number of test points
train_val_Y_2 = train_val_Y_2[:int(percentage*num_data2)]#take part of the data from second floor
train_val_X_2 = train_val_X_2[:int(percentage*num_data2)]

#Preprocessing
train_val_X_1 = preprocessing.scale(train_val_X_1) #zero mean
train_val_X_2 = preprocessing.scale(train_val_X_2)
test_X = preprocessing.scale(test_X)

#split training and validation data
train_X_1, val_X_1, train_Y_1, val_Y_1 = train_test_split(train_val_X_1, train_val_Y_1, test_size=0.3)
train_X_2, val_X_2, train_Y_2, val_Y_2 = train_test_split(train_val_X_2, train_val_Y_2, test_size=0.3)

#draw the graph of two floors
x,y = train_val_Y_1.T
plt.scatter(x,y)
c,d = train_val_Y_2.T
plt.scatter(c,d)
plt.show()

#neuron network without model transfer
# parameters
num_input = train_X_2.shape[1]# input layer size
act_fun = 'relu'
regularzation_penalty = 0.08
initilization_method = 'he_normal' #'random_uniform' ,'random_normal','TruncatedNormal' ,'glorot_uniform', 'glorot_nomral', 'he_normal', 'he_uniform'
#Optimizer
adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

#define model
model = Sequential()
model.add(Dense(500, activation=act_fun, input_dim=num_input, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
model.add(Dropout(0.5))
model.add(Dense(500, activation=act_fun, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
model.add(Dropout(0.5))
model.add(Dense(500, activation=act_fun, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='linear', kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
#Model compile
model.compile(loss='mean_squared_error',
              optimizer=adam)
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto')
#training with data
model.fit(train_X_2, train_Y_2,
          epochs=2000,
          batch_size=64,callbacks=[earlyStopping],validation_data=(val_X_2, val_Y_2))
train_loss_2_without_extrapolation = model.evaluate(train_X_2,train_Y_2, batch_size=len(train_Y_2))
val_loss_2_without_extrapolation= model.evaluate(val_X_2, val_Y_2, batch_size=len(val_Y_2))
test_loss_2_without_extrapolation = model.evaluate(test_X, test_Y, batch_size=len(test_Y))
predict_Y_without_extrapolation = model.predict(test_X)
#Neural network evaluation
error_NN_without_extrapolation = [None] * num_test #error vector for neural network
for i in range(num_test):
    error_NN_without_extrapolation[i] = np.linalg.norm(predict_Y_without_extrapolation[i] - test_Y[i])


#neuron network for model transfer
#define model
model = Sequential()
model.add(Dense(500, activation=act_fun, input_dim=num_input, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
model.add(Dropout(0.5))
model.add(Dense(500, activation=act_fun, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
model.add(Dropout(0.5))
model.add(Dense(500, activation=act_fun, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='linear', kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
#Model compile
model.compile(loss='mean_squared_error',
              optimizer=adam)
#pretrain with first floor's data
model.fit(train_X_1, train_Y_1,
          epochs=2000,
          batch_size=64,callbacks=[earlyStopping],validation_data=(val_X_1, val_Y_1))#tbCallBack,
train_loss_2_before = model.evaluate(train_X_2,train_Y_2, batch_size=len(train_Y_2)) #loss for training data2(not trained by neural network)
val_loss_2_before = model.evaluate(val_X_2, val_Y_2, batch_size=len(val_Y_2))
test_loss_2_before = model.evaluate(test_X, test_Y, batch_size=len(test_Y))
predict_Y_before = model.predict(test_X)
#situation 2
error_NN = [None] * num_test #error vector for neural network
for i in range(num_test):
    error_NN[i] = np.linalg.norm(predict_Y_before[i] - test_Y[i])

#fine tuning with the second floor's data
model.fit(train_X_2, train_Y_2,
          epochs=2000,
          batch_size=64,callbacks=[earlyStopping],validation_data=(val_X_2, val_Y_2))
predict_Y_extrapolation = model.predict(test_X)
train_loss_2_after = model.evaluate(train_X_2,train_Y_2, batch_size=len(train_Y_2)) #loss for training data2(not trained by neural network)
val_loss_2_after = model.evaluate(val_X_2, val_Y_2, batch_size=len(val_Y_2))
test_loss_2_after = model.evaluate(test_X, test_Y, batch_size=len(test_Y))
#situation 3
error_NN_extrapolation = [None] * num_test #error vector for neural network
for i in range(num_test):
    error_NN_extrapolation[i] = np.linalg.norm(predict_Y_extrapolation[i] - test_Y[i])

print("\nLoss for training data2 without extrapolation is",train_loss_2_without_extrapolation)
print("Loss for validation data2 without extrapolation is",val_loss_2_without_extrapolation)
print("Loss for test data without extrapolation is",test_loss_2_without_extrapolation)
print("\nLoss for training data2 before extrapolation is",train_loss_2_before)
print("Loss for validation data2 before extrapolation is",val_loss_2_before)
print("Loss for test data before extrapolation is",test_loss_2_before)
print("\nLoss for training data after fine tuning is",train_loss_2_after)
print("Loss for validation data after fine tuning is",val_loss_2_after)
print("Loss for test data after fine tuning is",test_loss_2_after)

#display
print('\nThe average error without extrapolation is',np.mean(error_NN_without_extrapolation),
      'minimum error:', np.amin(error_NN_without_extrapolation), 'maximum error:', np.amax(error_NN_without_extrapolation), 'variance:', np.var(error_NN_without_extrapolation) )
print('The average error before finetuning is',np.mean(error_NN),
      'minimum error:', np.amin(error_NN), 'maximum error:', np.amax(error_NN), 'variance:', np.var(error_NN) )
print('The average error after finetuning is',np.mean(error_NN_extrapolation),
      'minimum error:', np.amin(error_NN_extrapolation), 'maximum error:', np.amax(error_NN_extrapolation), 'variance:', np.var(error_NN_extrapolation))
plt.boxplot([error_NN_without_extrapolation, error_NN, error_NN_extrapolation ])# error_svm, , error_NN
plt.xticks([1, 2, 3], ['Original','Before fine tuning', 'After fine tuning'])#, 'Support Vector Machine'
plt.show()