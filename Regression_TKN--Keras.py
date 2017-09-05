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

# parameter
num_ED_knn = 3#k nearest neighbours
num_SVM_knn = 3
SVM_gamma = "auto"#1/num_feature "auto"
SVM_C = 1

#read training data
dataset = pd.read_csv("TKN_training_data.csv",header = 0)
rss_values = np.asarray(dataset.ix[:,0:-3]) #extract the RSSI values
train_val_Y = np.asarray(dataset.ix[:,-3:-1]) #extract the locations
train_val_X = np.asarray(rss_values, dtype=np.float64) #convert to numpy array
train_val_class = np.asarray(dataset.ix[:,-1]) #extract the class information for SVM

#reading test data
test_dataset = pd.read_csv("TKN_test_data.csv",header = 0)
rss_values_test = np.asarray(test_dataset.ix[:,0:-2])
test_Y= np.asarray(test_dataset.ix[:,-2:])
test_X = np.asarray(rss_values_test, dtype=np.float64)
a = len(train_val_X) #number of training points
b = len(test_X) #number of test points

#Preprocessing
train_val_X = preprocessing.scale(train_val_X) #zero mean
test_X = preprocessing.scale(test_X)

#split training and validation data
train_X, val_X, train_Y, val_Y = train_test_split(train_val_X, train_val_Y, test_size=0.3)

#draw the training grid
x,y = train_val_Y.T
plt.scatter(x,y)
plt.show()

#Euclidean distance-based method
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
    nearest_neighbour_ED[i] = np.argpartition(fingerprint_distance[i], num_ED_knn)[:num_ED_knn] #find the index of largest neighbours
    estimated_position_ED = np.mean(train_val_Y[nearest_neighbour_ED[i]], axis=0) #mean of all the neighbour positions
    error_ED[i] = np.linalg.norm(estimated_position_ED - test_Y[i])

# Support Vector Machine
#define SVM classifier
SVM = svm.SVC(decision_function_shape='ovr',kernel='rbf',gamma=SVM_gamma, C=SVM_C) #non-linear svm
SVM.fit(train_val_X, train_val_class) #feed samples and labels
test_score = SVM.decision_function(test_X) #scores for each test point n_test * n_training

#knn method combined with SVM
error_svm = [None] * num_test
nearest_neighbour_SVM = [None] * num_test
for i in range(num_test): #iteration of all test points
    nearest_neighbour_SVM[i] = np.argpartition(test_score[i], -num_SVM_knn)[-num_SVM_knn:] #find the index of largest score
    estimated_position_svm = np.mean(train_val_Y[10*nearest_neighbour_SVM[i]], axis=0) #mean of all the neighbour positions
    error_svm[i] = np.linalg.norm(estimated_position_svm - test_Y[i])

# neuron network regressor part using Keras
# train_X = train_val_X
# train_Y = train_val_Y
#neural network parameter
num_input = train_X.shape[1]# input layer size
act_fun = 'relu'
regularzation_penalty = 0.15
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
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=300, verbose=0, mode='auto')
model.fit(train_X, train_Y,
          epochs=10000,
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
print('The average error using Euclidean Distance:', np.mean(error_ED),
      'minimum error:',np.amin(error_ED),'maximum error:',np.amax(error_ED),'variance:',np.var(error_ED))
print('The average error using SVM is',np.mean(error_svm),
      'minimum error:',np.amin(error_svm), 'maximum error:', np.amax(error_svm),'variance:', np.var(error_svm))
print('The average error using NN regression is',np.mean(error_NN),
      'minimum error:', np.amin(error_NN), 'maximum error:', np.amax(error_NN), 'variance:', np.var(error_NN) )#
plt.boxplot([error_ED, error_svm, error_NN ])# error_svm, , error_NN
plt.xticks([1, 2, 3], ['Euclidean Distance','Support Vector Machine', 'Neural Network'])#, 'Support Vector Machine'
plt.show()