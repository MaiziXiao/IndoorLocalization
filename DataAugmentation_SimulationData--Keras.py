import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras import regularizers

#number of permutation of three situations
num_permutation_1 = 1
num_permutation_2 = 5
num_permutation_3 = 10

#parameter
width = 20 #size of the room
length = 10
num_test = 1000 #number of the random test numbers
num_measurements =5

#position of the access points
acpoint = [[0,0] , [20,0] , [0,10] , [20,10]]
n_acpoint = len(acpoint)

#parameter of the pass loss model
L0 = 40.22 #constant path loss at 1m distance and at center frequency of 2.45 Ghz
r = 1.64 #path loss exponent`
Lc = 53.73 #constant factor multi-wall model
k = 10 #number of walls
Lw = 4.51 #attenuation due to a wall
sigma = 0.000000000005 #parameter for normal distribution for noise
multipath_fading = 2 #parameter for multipath fading
transmitted_power = 20 #unit in dBm

#channel model:input is distance, output is RSS in unit mW
def multiwall_model(d): #multiwall model with respect to distance
    path_loss=L0 + 10*r*np.log10(d) + Lc + k*Lw  - 10*np.log10(np.random.exponential(multipath_fading))
    received_power = transmitted_power - path_loss#np.power(10, (transmitted_power - path_loss) / 10) #+ np.random.normal(0, sigma)
    return received_power
multiwall_model = np.vectorize(multiwall_model)

#trainning examples set up
#set-up for the trainning grid(square)
gap = 1 #distance between two neighbours trainning point
noise = gap / 4 #can make sure that it belongs to the same class
start_point = [gap/2, gap/2] #first left top point
i = start_point[0]
j = start_point[1]
Y_training_exact = [] #store the exact position of the points
Y_training_approximate = [] #store the approximated position of the points
Y_traininggrid = [] #store the position of those classes(only the middle point of each class)

#create exact position and approximate position of points
while j < length:
    while i < width:
        multiple_measurements = [[i,j],[i - noise, j - noise],[i + noise, j - noise],[i - noise, j + noise],[i + noise, j + noise]]
        Y_traininggrid.extend([[i,j]])
        Y_training_exact.extend(multiple_measurements)
        Y_training_approximate.extend([[i,j]] * len(multiple_measurements))
        i = i + gap
    j = j + gap
    i = start_point[0]
num_per_class = int(len(Y_training_exact) / len(Y_traininggrid)) #calculate the points per class
Y_training_exact = np.array(Y_training_exact)
Y_traininggrid = np.array(Y_traininggrid)
Y_training_approximate = np.array(Y_training_approximate)

#calculate the distance between trainning grids and access points distance[n][k]: k is the index of acess point, n is the index of the point
distance = np.zeros((len(Y_training_exact) ,n_acpoint))
for i in range(len(Y_training_exact)): #each point
    for j in range(len(acpoint)):
        distance[i][j] = np.linalg.norm(Y_training_exact[i] - acpoint[j])

#calulate the RSS(fingerprint) for each trainning grid point (unit in mW), each fingerprint include num_trainning * num_acpoint values
X_training = [None] * len(Y_training_exact) #fingerprint for neuron network
for i in range(num_measurements):
    temp = multiwall_model(distance)
    if i == 0:
        X_training = temp
    else:
        X_training = np.concatenate((X_training, temp), axis=1) #concatenate numbers of measurements for neural network

#set up test data
# pick random test points
Y_test = np.random.rand(num_test, 2)
for i in range(num_test):
    Y_test[i][0] = Y_test[i][0] * width
    Y_test[i][1] = Y_test[i][1] * length

# calculate the distance between test points and access points
distance_test = np.zeros((num_test, len(acpoint)))
for i in range(num_test):
    for j in range(len(acpoint)):
        distance_test[i][j] = np.linalg.norm(Y_test[i] - acpoint[j])

# calculate the RSSs(fingerprint) for each test point (unit in mW)
X_test = [None] * num_test
for i in range(num_measurements):
    temp = multiwall_model(distance_test)
    if i == 0:
        X_test = temp
    else:
        X_test = np.concatenate((X_test, temp), axis=1)


#permutation
#permutation function
def permute_columns(x):
    ix_i = np.random.sample(x.shape).argsort(axis=0)
    ix_j = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))
    return x[ix_i, ix_j]
#three levels of data augmentation
X_permutation_1 = X_training
X_permutation_2 = X_training
X_permutation_3 = X_training
#permutation of original data
for i in range(num_permutation_1):
    for j in range(len(Y_traininggrid)):
        X_permutation_1 = np.concatenate((X_permutation_1,permute_columns(X_training[j*num_per_class:j*num_per_class+num_per_class])))
    Y_permutation_1 = np.tile(Y_training_approximate, (num_permutation_1+1, 1))
for i in range(num_permutation_2):
    for j in range(len(Y_traininggrid)):
        X_permutation_2 = np.concatenate((X_permutation_2,permute_columns(X_training[j*num_per_class:j*num_per_class+num_per_class])))
    Y_permutation_2 = np.tile(Y_training_approximate, (num_permutation_2+1, 1))
for i in range(num_permutation_3):
    for j in range(len(Y_traininggrid)):
        X_permutation_3 = np.concatenate((X_permutation_3,permute_columns(X_training[j*num_per_class:j*num_per_class+num_per_class])))
    Y_permutation_3 = np.tile(Y_training_approximate, (num_permutation_3+1, 1))

#NN parameter
n_input = n_acpoint * num_measurements # input layer size
act_fun = 'relu'
regularzation_penalty = 0.03
initilization_method = 'he_normal' #'random_uniform' ,'random_normal','TruncatedNormal' ,'glorot_uniform', 'glorot_nomral', 'he_normal', 'he_uniform'
#Optimizer
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto')

def model_build():
    model.add(Dense(100, activation=act_fun, input_dim=n_input, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
    # model.add(Dropout(0.5))
    model.add(Dense(100, activation=act_fun, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
    # model.add(Dropout(0.5))
    model.add(Dense(100, activation=act_fun, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
    # model.add(Dropout(0.5))
    model.add(Dense(2, activation='linear', kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))

# without permutation
model = Sequential()
model_build()
#Model compile
model.compile(loss='mean_squared_error',
              optimizer=adam)
model.fit(X_training, Y_training_approximate,
          epochs=1000,
          batch_size=200,callbacks=[earlyStopping],validation_data=(X_test, Y_test)) #tbCallBack,
test_loss = model.evaluate(X_test, Y_test, batch_size=len(X_test))
Y_predict = model.predict(X_test)
#Calculate error
error_NN_regression = [None] * num_test
for i in range(num_test):
    error_NN_regression[i] = np.linalg.norm(Y_predict[i] - Y_test[i])

#with permutation1
model = Sequential()
model_build()
#Model compile
model.compile(loss='mean_squared_error',
              optimizer=adam)
model.fit(X_permutation_1, Y_permutation_1,
          epochs=1000,
          batch_size=200,callbacks=[earlyStopping],validation_data=(X_test, Y_test)) #tbCallBack,
test_loss = model.evaluate(X_permutation_1, Y_permutation_1, batch_size=len(X_test))
Y_predict_1 = model.predict(X_test)
#Calculate error
error_NN_permutation_1 = [None] * num_test
for i in range(num_test):
    error_NN_permutation_1[i] = np.linalg.norm(Y_predict_1[i] - Y_test[i])

#with permutation_2
model = Sequential()
model_build()
#Model compile
model.compile(loss='mean_squared_error',
              optimizer=adam)
model.fit(X_permutation_2, Y_permutation_2,
          epochs=1000,
          batch_size=200,callbacks=[earlyStopping],validation_data=(X_test, Y_test)) #tbCallBack,
test_loss = model.evaluate(X_permutation_2, Y_permutation_2, batch_size=len(X_test))
Y_predict_2 = model.predict(X_test)
#Calculate error
error_NN_permutation_2 = [None] * num_test
for i in range(num_test):
    error_NN_permutation_2[i] = np.linalg.norm(Y_predict_2[i] - Y_test[i])

#with permutation_3
model = Sequential()
model_build()
#Model compile
model.compile(loss='mean_squared_error',
              optimizer=adam)
model.fit(X_permutation_3, Y_permutation_3,
          epochs=1000,
          batch_size=200,callbacks=[earlyStopping],validation_data=(X_test, Y_test)) #tbCallBack,
test_loss = model.evaluate(X_permutation_3, Y_permutation_3, batch_size=len(X_test))
Y_predict_3 = model.predict(X_test)
#Calculate error
error_NN_permutation_3 = [None] * num_test
for i in range(num_test):
    error_NN_permutation_3[i] = np.linalg.norm(Y_predict_3[i] - Y_test[i])

#display
print('\nThe average error of original data is',np.mean(error_NN_regression),'meter','minimum error:',np.amin(error_NN_regression),'maximum error:',np.amax(error_NN_regression),'variance:',np.var(error_NN_regression))
print('The average error of data augmentation 1 is',np.mean(error_NN_permutation_1),'meter','minimum error:',np.amin(error_NN_permutation_1),'maximum error:',np.amax(error_NN_permutation_1),'variance:',np.var(error_NN_permutation_1))
print('The average error of data augmentation 2 is',np.mean(error_NN_permutation_2),'meter','minimum error:',np.amin(error_NN_permutation_2),'maximum error:',np.amax(error_NN_permutation_2),'variance:',np.var(error_NN_permutation_2))
print('The average error of data augmentation 3 is',np.mean(error_NN_permutation_3),'meter','minimum error:',np.amin(error_NN_permutation_3),'maximum error:',np.amax(error_NN_permutation_3),'variance:',np.var(error_NN_permutation_3))
