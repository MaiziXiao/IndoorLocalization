import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import svm

#parameter
width = 20 #size of the room
length = 10
num_test = 1000 #number of the random test numbers
num_measurements = 5 #number of measurements
num_kNN = 3 #k nearest neighbour parameter
svm_C = 1.0# SVM parameter
svm_gamma = 'auto'#SVM parameter

#parameters for neuron network
learning_rate =1
training_epochs = 10001
batch_size = 50
n_hidden = 5#single hidden layer size

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
    received_power = transmitted_power - path_loss
    return received_power
multiwall_model = np.vectorize(multiwall_model)

#trainning examples set up
#set-up for the trainning grid(square)
distance = 1 #distance between two neighbours trainning point
noise = distance / 4 #can make sure that it belongs to the same class
start_point = [distance/2, distance/2] #first left top point
i = start_point[0]
j = start_point[1]
data = []
localization_class = []
while j < length:
    while i < width:
        localization_class.extend([[i,j]])
        data.extend([[i,j],[i - noise, j - noise],[i + noise, j - noise],[i - noise, j + noise],[i + noise, j + noise]])
        i = i + distance
    j = j + distance
    i = start_point[0]
num_per_class = int(len(data) / len(localization_class))
data = np.array(data) #convert to numpy array
localization_class = np.array(localization_class)

#create labels for each class
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

#test data set up
# pick random test points
test_position = np.random.rand(num_test, 2)
for i in range(num_test):
    test_position[i][0] = test_position[i][0] * width
    test_position[i][1] = test_position[i][1] * length

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

#Euclidean distance-based method
#calculate the Euclidean distance of fingerprints of all test points and trainning grids.
#fingerprint_distance[n][k]: n indicates which test point, k indicates which trainning point
fingerprint_distance_traditional = np.zeros((num_test,len(data)))
for i in range(num_test):
    for j in range(len(data)):
        fingerprint_distance_traditional[i][j] = np.linalg.norm(fingerprint_traditional[j]-fingerprint_test_traditional[i])

# kNN method
nsmallest_distance_traditional_index = [None] * num_test
estimated_position_knn_traditional = [None] * num_test
error_knn_traditional = [None] * num_test
for i in range(num_test):  # iteration of all test points
    nsmallest_distance_traditional_index[i] = np.argpartition(fingerprint_distance_traditional[i], num_kNN)[:num_kNN] #find the index of largest neighbours
    estimated_position_knn_traditional[i] = np.mean(data[nsmallest_distance_traditional_index[i]], axis=0) #mean of all the neighbour positions
    error_knn_traditional[i] = np.linalg.norm(estimated_position_knn_traditional[i] - test_position[i]) #error between estimated and true target

#Support Vector Machine
#normalization
min = fingerprint_test_traditional.min()
max = fingerprint_test_traditional.max()
fingerprint_traditional = (fingerprint_traditional - min)/(max-min) #normalize
fingerprint_test_traditional = (fingerprint_test_traditional - min)/(max-min) #normalize
SVM_traditional = svm.SVC(decision_function_shape='ovr',kernel='rbf',gamma=svm_gamma, C=svm_C)
SVM_traditional.fit(fingerprint_traditional, Y) #feed samples and labels
test_prediction = SVM_traditional.predict(fingerprint_test_traditional)
test_score = SVM_traditional.decision_function(fingerprint_test_traditional)

#knn method combined with SVM
nlargest_score_traditional_index = [None] * num_test #For each test sample, which class candidate it has
estimated_position_svm_knn_traditional = [None] * num_test
error_svm_knn_traditional = [None] * num_test
for i in range(num_test): #iteration of all test points
    nlargest_score_traditional_index[i] = np.argpartition(test_score[i], -num_kNN)[-num_kNN:] #find the index of largest neighbours
    estimated_position_svm_knn_traditional[i] = np.mean(localization_class[nlargest_score_traditional_index[i]], axis=0) #mean of all the neighbour positions
    error_svm_knn_traditional[i] = np.linalg.norm(estimated_position_svm_knn_traditional[i] - test_position[i])

#Autoencoder using tensorflow
#normalize the data
min = fingerprint_test.min()
max = fingerprint_test.max()
fingerprint = (fingerprint - min)/(max-min)
fingerprint_test = (fingerprint_test - min)/(max-min)

#parameters
total_batch = int(fingerprint.shape[0]/ batch_size)
n_input = n_acpoint * num_measurements # input layer size
trainning_data =np.array(fingerprint) #copy for mini-batch trainning(shuffle data)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

#weights and biases of neuorn network
init = np.sqrt(6. / (n_input + n_hidden))
encoder_h1 = tf.Variable(tf.random_uniform([n_input,n_hidden], -1*init, init)) #weight matrix of encoder
decoder_h1 = tf.Variable(tf.random_uniform([n_hidden, n_input], -1*init, init)) #weight matrix of decoder
encoder_b1 = tf.Variable(tf.zeros([n_hidden])) #biases vector of encoder
decoder_b1 = tf.Variable(tf.zeros([n_input]))  #biases vector of decoder

#encoder and decoder function
def encoder(x):
    hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(x, encoder_h1),encoder_b1) )
    return hidden_layer
def decoder(x):
    output_layer =tf.nn.sigmoid( tf.add(tf.matmul(x, decoder_h1),decoder_b1) )
    return output_layer
# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.square(y_true - y_pred))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Launch the graph in a session
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(training_epochs):
        # Loop over all batches
        np.random.shuffle(trainning_data)#shuffle the data
        for i in range(total_batch):
            sess.run(optimizer, {X: trainning_data[i * batch_size: (i+1)*batch_size]}) #feed data
        if epoch%1000==0:
            print("Epoch:",epoch ,"Cost for test data:",sess.run(cost, {X: fingerprint_test})) #used test data to calculate the cost
    print("Cost for trainning data:", sess.run(cost, {X:fingerprint}))
    fingerprint = sess.run(encoder(X), {X:fingerprint}) #get the encoded fingerprint
    fingerprint_test = sess.run(encoder(X), {X:fingerprint_test})

#Eculidean distance on encoded fingerprint
#calculate the Euclidean distance of fingerprints of all test points and trainning grids.
# fingerprint_distance[n][k]: n indicates which test point, k indicates which trainning point
fingerprint_distance = np.zeros((num_test,len(data)))
for i in range(num_test):
    for j in range(len(data)):
        fingerprint_distance[i][j] = np.linalg.norm(fingerprint[j]-fingerprint_test[i])

#kNN method
nsmallest_distance_index = [None] * num_test
estimated_position_knn = [None] * num_test
error_knn = [None] * num_test
for i in range(num_test): #iteration of all test points
    nsmallest_distance_index[i] = np.argpartition(fingerprint_distance[i], num_kNN)[:num_kNN] #find the index of largest neighbours
    estimated_position_knn[i] = np.mean(data[nsmallest_distance_index[i]], axis=0) #mean of all the neighbour positions
    error_knn[i] = np.linalg.norm(estimated_position_knn[i] - test_position[i])

#SVM on encoded fingerprint
clf = svm.SVC(decision_function_shape='ovr',kernel='rbf',gamma=svm_gamma, C=svm_C)
clf.fit(fingerprint, Y) #feed samples and labels
test_prediction = clf.predict(fingerprint_test)
test_score = clf.decision_function(fingerprint_test)

#k-nearest neighbour for svm
nlargest_score_index = [None] * num_test #For each test sample, which class candidate it has
estimated_position_svm_knn = [None] * num_test
error_svm_knn = [None] * num_test
for i in range(num_test): #iteration of all test points
    nlargest_score_index[i] = np.argpartition(test_score[i], -num_kNN)[-num_kNN:] #find the index of largest neighbours
    estimated_position_svm_knn[i] = np.mean(localization_class[nlargest_score_index[i]], axis=0) #mean of all the neighbour positions
    error_svm_knn[i] = np.linalg.norm(estimated_position_svm_knn[i] - test_position[i])

#display the result
print('The average error using KNN(without neural network) is', np.mean(error_knn_traditional), 'meter',
      'amin error:', np.amin(error_knn_traditional), 'amax error:', np.amax(error_knn), 'variance:',np.var(error_knn_traditional))
print('The average error using SVM(withour neural network) is',np.mean(error_svm_knn_traditional),'meter',
      'amin error:', np.amin(error_svm_knn_traditional), 'amax error:', np.amax(error_svm_knn_traditional), 'variance:',np.var(error_svm_knn_traditional))
print('The average error using KNN is',np.mean(error_knn),'meter',
      'amin error:', np.amin(error_knn), 'amax error:', np.amax(error_knn), 'variance:',np.var(error_knn))
print('The average error using SVM(KNN) is',np.mean(error_svm_knn),'meter',
      'amin error:', np.amin(error_svm_knn), 'amax error:', np.amax(error_svm_knn), 'variance:', np.var(error_svm_knn))
plt.boxplot([error_knn_traditional,error_svm_knn_traditional, error_knn, error_svm_knn])
plt.show()

