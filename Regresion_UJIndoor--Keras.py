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

# SVM parameter
SVM_knn = 3
SVM_gamma = "auto"#1/num_feature
SVM_C = 7
ED_knn = 3#k nearest neighbours

#read training data
dataset = pd.read_csv("UJIndoor_training_data.csv",header = 0)
dataset =  dataset[(dataset['FLOOR'] == FloorID)  & (dataset['BUILDINGID'] == BuildingID) ] #select the floor and building  & (dataset['BUILDINGID'] == BuildingID)
rss_values = np.asarray(dataset.ix[:,0:-9]) #extract the features(delete the location, floor information etc.)
rss_values[rss_values == 100] = -110 #make undefined measurements -110dbm
locations = np.asarray(dataset.ix[:,-9:-7])
origin =np.amin(locations,axis=0) #calculate the origin
room_size = np.amax(locations, axis=0) - origin #size of the room
train_val_Y = locations - origin #position respect to origin
train_val_X = np.asarray(rss_values, dtype=np.float64) #convert to numpy array

#reading test data
test_dataset = pd.read_csv("UJIndoor_test_data.csv",header = 0)
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
train_val_class = np.zeros(len(train_val_Y))#create the array to store class of training data
num_unique_position = len(unique_position) #how many points in training grid
for i in range(num_unique_position): #for each point in training grid
    in_this_class = train_val_Y[:] == unique_position[i] #find the index which has the same position as the training grid
    in_this_class = in_this_class[:,0]
    train_val_class[in_this_class]= i #label them
    sample_in_this_class = train_val_X[in_this_class]  # training sample with same location (prepared for permutation)

#split training and validation data\
train_X, val_X, train_Y, val_Y = train_test_split(train_val_X, train_val_Y, test_size=0.3)
#draw the position of training points in this floor
x,y = train_val_Y.T
plt.scatter(x,y)
plt.show()

#Euclidean distance
#calculate the Euclidean distance of fingerprints of all test points and trainning grids.
# fingerprint_distance[n][k]: n indicates which test point, k indicates which trainning point
num_test = test_X.shape[0]
fingerprint_distance= np.zeros((num_test,len(train_val_X)))
for i in range(num_test):
    for j in range(len(train_val_X)):
        fingerprint_distance[i][j] = np.linalg.norm(train_val_X[j]-test_X[i])
error_ED = [None] * num_test
nearest_neighbour_ED = [None] * num_test
for i in range(num_test):  # iteration of all test points
    nearest_neighbour_ED[i] = np.argpartition(fingerprint_distance[i], ED_knn)[:ED_knn] #find the index of largest neighbours
    estimated_position_ED = np.mean(train_val_Y[nearest_neighbour_ED[i]], axis=0) #mean of all the neighbour positions
    error_ED[i] = np.linalg.norm(estimated_position_ED - test_Y[i])

# Support Vector Machine
#SVM training
SVM = svm.SVC(decision_function_shape='ovr',kernel='rbf',gamma=SVM_gamma, C=SVM_C) #non-linear svm
SVM.fit(train_val_X, train_val_class) #feed samples and labels
test_score = SVM.decision_function(test_X) #scores for each test point n_test * n_training
#knn method combined with SVM
error_svm = [None] * num_test
nearest_neighbour_SVM = [None] * num_test
for i in range(num_test): #iteration of all test points
    nearest_neighbour_SVM[i] = np.argpartition(test_score[i], -SVM_knn)[-SVM_knn:] #find the index of largest score
    estimated_position_svm = np.mean(unique_position[nearest_neighbour_SVM[i]], axis=0) #mean of all the neighbour positions
    error_svm[i] = np.linalg.norm(estimated_position_svm - test_Y[i])

#neuron network regressor part using Keras
# parameters
num_input = train_X.shape[1]# input layer size
act_fun = 'relu'
regularzation_penalty = 0.03
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
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, verbose=0, mode='auto')
model.fit(train_X, train_Y,
          epochs=1000,
          batch_size=64,callbacks=[earlyStopping],validation_data=(val_X, val_Y))#tbCallBack,
train_loss = model.evaluate(train_X,train_Y, batch_size=len(train_Y)) #calculate the data in test mode(Keras)
val_loss = model.evaluate(val_X, val_Y, batch_size=len(val_Y))
test_loss = model.evaluate(test_X, test_Y, batch_size=len(test_Y))
print("Loss for training data is",train_loss)
print("Loss for validation data is",val_loss)
print("Loss for test data is",test_loss)
predict_Y = model.predict(test_X)

#Neural network evaluation
error_NN = [None] * num_test #error vector for neural network
for i in range(num_test):
    error_NN[i] = np.linalg.norm(predict_Y[i] - test_Y[i])

##display the result
print('\nThe room size is:', room_size)
print(a,'training points',b,'test points')
print('The average error using Euclidean Distance:', np.mean(error_ED),
      'minimum error:',np.amin(error_ED),'maximum error:',np.amax(error_ED),'variance:',np.var(error_ED))
print('The average error using SVM is',np.mean(error_svm),
      'minimum error:',np.amin(error_svm), 'maximum error:', np.amax(error_svm),'variance:', np.var(error_svm))
print('The average error using NN regression is',np.mean(error_NN),
      'minimum error:', np.amin(error_NN), 'maximum error:', np.amax(error_NN), 'variance:', np.var(error_NN) )#
plt.boxplot([error_ED, error_svm, error_NN ])# error_svm, , error_NN
plt.xticks([1, 2, 3], ['Euclidean Distance','Support Vector Machine', 'Neural Network'])#, 'Support Vector Machine'
plt.show()