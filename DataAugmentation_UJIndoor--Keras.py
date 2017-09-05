import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
import matplotlib.pyplot as plt
from keras import regularizers
import sklearn
from sklearn import svm
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

#dataset parameter
BuildingID =0
FloorID =0

#number of permutation of three situations
num_permutation_1 = 1
num_permutation_2 = 3
num_permutation_3 = 5

#read training data
dataset = pd.read_csv("trainingData.csv",header = 0)
dataset =  dataset[(dataset['FLOOR'] == FloorID)  & (dataset['BUILDINGID'] == BuildingID) ] #select the floor and building  & (dataset['BUILDINGID'] == BuildingID)
rss_values = np.asarray(dataset.ix[:,0:-9]) #extract the features(delete the location, floor information etc.)
rss_values[rss_values == 100] = -110 #make undefined measurements -110dbm
locations = np.asarray(dataset.ix[:,-9:-7])
origin =np.amin(locations,axis=0) #calculate the origin
room_size = np.amax(locations, axis=0) - origin #size of the room
train_val_Y = locations - origin #position respect to origin
train_val_X = np.asarray(rss_values, dtype=np.float64) #convert to numpy array

#reading test data
test_dataset = pd.read_csv("validationData.csv",header = 0)
test_dataset =  test_dataset[(test_dataset['FLOOR'] == FloorID) & (test_dataset['BUILDINGID'] == BuildingID) ]  #& (test_dataset['BUILDINGID'] == BuildingID)
rss_values_test = np.asarray(test_dataset.ix[:,0:-9])
rss_values_test[rss_values_test == 100] = -110
test_locations= np.asarray(test_dataset.ix[:,-9:-7])
test_Y = test_locations - origin
test_X = np.asarray(rss_values_test, dtype=np.float64)
a = len(train_val_X) #number of training points
b = len(test_X) #number of test points

#Preprocessing
train_val_X = preprocessing.scale(train_val_X) #zero mean
test_X = preprocessing.scale(test_X)

#find training position in the training data and permutation
unique_position = np.vstack({tuple(row) for row in train_val_Y}) #find unique training grid
# train_val_class = np.zeros(len(train_val_Y))#create the array to store class of training data
num_unique_position = len(unique_position) #how many points in training grid
train_val_X_permutation_1 = np.array(train_val_X)  # copy the original data
train_val_Y_permutation_1 = np.array(train_val_Y)
train_val_X_permutation_2 = np.array(train_val_X)  # copy the original data
train_val_Y_permutation_2 = np.array(train_val_Y)
train_val_X_permutation_3 = np.array(train_val_X)  # copy the original data
train_val_Y_permutation_3 = np.array(train_val_Y)
for i in range(num_unique_position): #for each point in training grid
    in_this_class = train_val_Y[:] == unique_position[i] #find the index which has the same position as the training grid
    in_this_class = in_this_class[:,0]
    # train_val_class[in_this_class]= i #label them
    sample_in_this_class = train_val_X[in_this_class]  # training sample with same location (prepared for permutation)
    #permutation_1
    for j in range(num_permutation_1):
        for k in range(sample_in_this_class.shape[1]): #permute each column independently
            np.random.shuffle(sample_in_this_class[:,k])
        train_val_X_permutation_1 = np.concatenate((train_val_X_permutation_1, sample_in_this_class), axis=0)  # concatenate with the old training data
        train_val_Y_permutation_1 = np.concatenate((train_val_Y_permutation_1, np.tile(unique_position[i],(len(sample_in_this_class),1))), axis=0) #labels of permutation data
    # permutation_2
    for j in range(num_permutation_2): #permutation_2
        for k in range(sample_in_this_class.shape[1]): #permute each column independently
            np.random.shuffle(sample_in_this_class[:,k])
        train_val_X_permutation_2 = np.concatenate((train_val_X_permutation_2, sample_in_this_class), axis=0)  # concatenate with the old training data
        train_val_Y_permutation_2 = np.concatenate((train_val_Y_permutation_2, np.tile(unique_position[i],(len(sample_in_this_class),1))), axis=0) #labels of permutation data
    # permutation_3
    for j in range(num_permutation_3): #permutation_3
        for k in range(sample_in_this_class.shape[1]): #permute each column independently
            np.random.shuffle(sample_in_this_class[:,k])
        train_val_X_permutation_3 = np.concatenate((train_val_X_permutation_3, sample_in_this_class), axis=0)  # concatenate with the old training data
        train_val_Y_permutation_3 = np.concatenate((train_val_Y_permutation_3, np.tile(unique_position[i],(len(sample_in_this_class),1))), axis=0) #labels of permutation data


#split training and validation data\
train_X, val_X, train_Y, val_Y = train_test_split(train_val_X, train_val_Y, test_size=0.3)
train_X_permutation_1, val_X_permutation_1, train_Y_permutation_1, val_Y_permutation_1 = train_test_split(train_val_X_permutation_1, train_val_Y_permutation_1, test_size=0.3)
train_X_permutation_2, val_X_permutation_2, train_Y_permutation_2, val_Y_permutation_2 = train_test_split(train_val_X_permutation_2, train_val_Y_permutation_2, test_size=0.3)
train_X_permutation_3, val_X_permutation_3, train_Y_permutation_3, val_Y_permutation_3 = train_test_split(train_val_X_permutation_3, train_val_Y_permutation_3, test_size=0.3)
#draw the position of training points in this floor
x,y = train_val_Y.T
plt.scatter(x,y)
plt.show()

#neuron network regressor part using Keras
#parameters
num_test = test_X.shape[0] #number of training examples
num_input = train_X.shape[1]# input layer size
act_fun = 'relu'
regularzation_penalty = 0.03
initilization_method = 'he_normal' #'random_uniform' ,'random_normal','TruncatedNormal' ,'glorot_uniform', 'glorot_nomral', 'he_normal', 'he_uniform'
num_batch = 64 #number of batch size
num_epoch = 1000 #number of epoch
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, verbose=0, mode='auto')
#Optimizer
adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

#define model
def model_build():
    model.add(Dense(500, activation=act_fun, input_dim=num_input, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation=act_fun, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation=act_fun, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='linear', kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))

#without permutation
model = Sequential()
model_build()
#Model compile
model.compile(loss='mean_squared_error',
              optimizer=adam)
model.fit(train_X, train_Y,
          epochs=num_epoch,
          batch_size=num_batch,callbacks=[earlyStopping],validation_data=(val_X, val_Y))#
predict_Y = model.predict(test_X)
#neural network evaluation
error_NN = [None] * num_test #error vector for neural network
for i in range(num_test):
    error_NN[i] = np.linalg.norm(predict_Y[i] - test_Y[i])

#permutation_1
model = Sequential()
model_build()
#Model compile
model.compile(loss='mean_squared_error',
              optimizer=adam)
model.fit(train_X_permutation_1, train_Y_permutation_1,
          epochs=num_epoch,
          batch_size=num_batch,callbacks=[earlyStopping],validation_data=(val_X_permutation_1, val_Y_permutation_1))#
predict_Y_permutation_1 = model.predict(test_X)
#neural network evaluation
error_NN_permutation_1 = [None] * num_test #error vector for neural network
for i in range(num_test):
    error_NN_permutation_1[i] = np.linalg.norm(predict_Y_permutation_1[i] - test_Y[i])

#permutation_2
model = Sequential()
model_build()
#Model compile
model.compile(loss='mean_squared_error',
              optimizer=adam)
model.fit(train_X_permutation_2, train_Y_permutation_2,
          epochs=num_epoch,
          batch_size=num_batch,callbacks=[earlyStopping],validation_data=(val_X_permutation_2, val_Y_permutation_2))#
predict_Y_permutation_2 = model.predict(test_X)
#neural network evaluation
error_NN_permutation_2 = [None] * num_test #error vector for neural network
for i in range(num_test):
    error_NN_permutation_2[i] = np.linalg.norm(predict_Y_permutation_2[i] - test_Y[i])

#permutation_3
model = Sequential()
model_build()
#Model compile
model.compile(loss='mean_squared_error',
              optimizer=adam)
model.fit(train_X_permutation_3, train_Y_permutation_3,
          epochs=num_epoch,
          batch_size=num_batch,callbacks=[earlyStopping],validation_data=(val_X_permutation_3, val_Y_permutation_3))#
predict_Y_permutation_3 = model.predict(test_X)
#neural network evaluation
error_NN_permutation_3 = [None] * num_test #error vector for neural network
for i in range(num_test):
    error_NN_permutation_3[i] = np.linalg.norm(predict_Y_permutation_3[i] - test_Y[i])

#display the result
print('\nThe room size is:', room_size)
print(a,'training points',b,'test points')
print('The average error of original data is',np.mean(error_NN),
      'minimum error:', np.amin(error_NN), 'maximum error:', np.amax(error_NN), 'variance:', np.var(error_NN) )
print('The average error of data with permutation_1 is',np.mean(error_NN_permutation_1),
      'minimum error:', np.amin(error_NN_permutation_1), 'maximum error:', np.amax(error_NN_permutation_1), 'variance:', np.var(error_NN_permutation_1) )
print('The average error of data with permutation_2 is',np.mean(error_NN_permutation_2),
      'minimum error:', np.amin(error_NN_permutation_2), 'maximum error:', np.amax(error_NN_permutation_2), 'variance:', np.var(error_NN_permutation_2) )
print('The average error of data with permutation_3 is',np.mean(error_NN_permutation_3),
      'minimum error:', np.amin(error_NN_permutation_3), 'maximum error:', np.amax(error_NN_permutation_3), 'variance:', np.var(error_NN_permutation_3) )
