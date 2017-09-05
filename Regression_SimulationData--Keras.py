import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras import regularizers
from sklearn import svm

#parameter
width = 20 #size of the room
length = 10
num_test = 1000 #number of the random test numbers
num_measurements =5
num_kNN = 5 #k nearest neighbour parameter
svm_C = 1.0# SVM parameter
svm_gamma = 'auto'#SVM parameter

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
gap = 1#distance between two neighbours trainning point
noise = gap / 4 #can make sure that it belongs to the same class
start_point = [gap/2, gap/2] #first left top point
i = start_point[0]
j = start_point[1]
data = [] #store the exact position of the points
data_approximate = [] #store the approximated position of the points
localization_class = [] #store the position of those classes(only the middle point of each class)

#create approximate position of points
while j < length:
    while i < width:
        multiple_measurements = [[i,j],[i - noise, j - noise],[i + noise, j - noise],[i - noise, j + noise],[i + noise, j + noise]]
        localization_class.extend([[i,j]])
        data.extend(multiple_measurements)
        data_approximate.extend([[i,j]] * len(multiple_measurements))
        i = i + gap
    j = j + gap
    i = start_point[0]
num_per_class = int(len(data) / len(localization_class)) #calculate the points per class
data = np.array(data)
localization_class = np.array(localization_class)
data_approximate = np.array(data_approximate)

#create label for each class
Y = []
for i in range(len(localization_class)):
    Y.extend([i] * num_per_class)

#calculate the distance between trainning grids and access points distance[n][k]: k is the index of acess point, n is the index of the point
distance = np.zeros((len(data) ,n_acpoint))
for i in range(len(data)): #each point
    for j in range(len(acpoint)):
        distance[i][j] = np.linalg.norm(data[i] - acpoint[j])

#calulate the RSS(fingerprint) for each trainning grid point (unit in mW), each fingerprint include num_trainning * num_acpoint values
fingerprint = [None] * len(data) #fingerprint for neuron network
fingerprint_traditional = [None] * len(data) #fingerprint without neuron network
for i in range(num_measurements):
    temp = multiwall_model(distance)
    if i == 0:
        fingerprint = temp
        fingerprint_traditional = temp
    else:
        fingerprint = np.concatenate((fingerprint, temp), axis=1) #concatenate numbers of measurements for neural network
        fingerprint_traditional = fingerprint_traditional + temp #add up measurements for traditional method
fingerprint_traditional = np.divide(fingerprint_traditional, num_measurements) #average with measument number

#set up test data
#pick random test points
test_position = np.zeros([num_test,2])
Y_test = [None] * num_test #labels for test data
for i in range(num_test):
    temp = np.random.randint(len(localization_class)) #pick one class from all classes
    Y_test[i] = temp #give label to the sample
    test_position[i][0] = localization_class[temp][0]+ np.random.uniform(-gap/2,gap/2)
    test_position[i][1] = localization_class[temp][1]+ np.random.uniform(-gap/2,gap/2)
# calculate the distance between test points and access points
distance_test = np.zeros((num_test, len(acpoint)))
for i in range(num_test):
    for j in range(len(acpoint)):
        distance_test[i][j] = np.linalg.norm(test_position[i] - acpoint[j])
# calculate the RSSs(fingerprint) for each test point (unit in mW)
fingerprint_test = [None] * num_test
fingerprint_test_traditional = [None] * num_test
for i in range(num_measurements):
    temp = multiwall_model(distance_test)
    if i == 0:
        fingerprint_test = temp
        fingerprint_test_traditional = temp
    else:
        fingerprint_test = np.concatenate((fingerprint_test, temp), axis=1)
        fingerprint_test_traditional = fingerprint_test_traditional + temp
fingerprint_test_traditional = np.divide(fingerprint_test_traditional, num_measurements)


#Euclidean distance_based method
#calculate the Euclidean distance of fingerprints of all test points and trainning grids.
# fingerprint_distance[n][k]: n indicates which test point, k indicates which trainning point
fingerprint_distance_traditional = np.zeros((num_test,len(data)))
for i in range(num_test):
    for j in range(len(data)):
        fingerprint_distance_traditional[i][j] = np.linalg.norm(fingerprint_traditional[j]-fingerprint_test_traditional[i])
error_knn_traditional = [None] * num_test
for i in range(num_test):  # iteration of all test points
    temp = np.argpartition(fingerprint_distance_traditional[i], num_kNN)[:num_kNN] #find the index of largest neighbours
    estimated_position_knn_traditional = np.mean(data_approximate[temp], axis=0) #mean of all the neighbour positions
    error_knn_traditional[i] = np.linalg.norm(estimated_position_knn_traditional - test_position[i])


#Support Vector Machine
#normalize
min = fingerprint_test_traditional.min()
max = fingerprint_test_traditional.max()
fingerprint_traditional = (fingerprint_traditional - min)/(max-min)
fingerprint_test_traditional = (fingerprint_test_traditional - min)/(max-min)
#define SVM classifier
SVM_traditional = svm.SVC(decision_function_shape='ovr',kernel='rbf',gamma=svm_gamma, C=svm_C)
SVM_traditional.fit(fingerprint_traditional, Y) #feed samples and labels
test_prediction = SVM_traditional.predict(fingerprint_test_traditional)
test_score = SVM_traditional.decision_function(fingerprint_test_traditional)

#knn method combined with SVM
error_svm_knn_traditional = [None] * num_test
for i in range(num_test): #iteration of all test points
    temp = np.argpartition(test_score[i], -num_kNN)[-num_kNN:] #find the index of largest neighbours
    estimated_position_svm_knn_traditional = np.mean(localization_class[temp], axis=0) #mean of all the neighbour positions
    error_svm_knn_traditional[i] = np.linalg.norm(estimated_position_svm_knn_traditional - test_position[i])


##neuron network
#normalize
min = fingerprint_test.min()
max = fingerprint_test.max()
fingerprint = (fingerprint - min)/(max-min)
fingerprint_test = (fingerprint_test - min)/(max-min)

#parameters
n_input = n_acpoint * num_measurements # input layer size
act_fun = 'relu'
regularzation_penalty = 0.03
initilization_method = 'he_normal' #'random_uniform' ,'random_normal','TruncatedNormal' ,'glorot_uniform', 'glorot_nomral', 'he_normal', 'he_uniform'
#Optimizer
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

#define model
model = Sequential()
model.add(Dense(500, activation=act_fun, input_dim=n_input, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
# model.add(Dropout(0.5))
model.add(Dense(500, activation=act_fun, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
# model.add(Dropout(0.5))
model.add(Dense(500, activation=act_fun, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
# model.add(Dropout(0.5))
model.add(Dense(2, activation='linear', kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))

#Model compile
model.compile(loss='mean_squared_error',
              optimizer=adam)
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto')
model.fit(fingerprint, data_approximate,
          epochs=1,
          batch_size=32,callbacks=[earlyStopping],validation_data=(fingerprint_test, test_position))#tbCallBack,
test_loss = model.evaluate(fingerprint_test, test_position, batch_size=len(fingerprint_test))
Y_regression = model.predict(fingerprint_test)

#neural network regression
error_NN_regression = [None] * num_test
for i in range(num_test):
    error_NN_regression[i] = np.linalg.norm(Y_regression[i] - test_position[i])

##display the result
print('\nThe average error using KNN(without neural network):', np.mean(error_knn_traditional),
      'minimum error:',np.amin(error_knn_traditional),'maximum error:',np.amax(error_knn_traditional),'variance:',np.var(error_knn_traditional))
print('The average error using SVM(withour neural network) is',np.mean(error_svm_knn_traditional),
      'minimum error:',np.amin(error_svm_knn_traditional), 'maximum error:', np.amax(error_svm_knn_traditional),'variance:', np.var(error_svm_knn_traditional))
print('The average error using NN regression is',np.mean(error_NN_regression),
      'minimum error:',np.amin(error_NN_regression), 'maximum error:', np.amax(error_NN_regression),'variance:', np.var(error_NN_regression))
plt.boxplot([error_knn_traditional,error_svm_knn_traditional,  error_NN_regression])
plt.xticks([1, 2, 3], ['Euclidean Distance', 'Support Vector Machine', 'Neural Network'])
plt.show()
