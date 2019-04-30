import tensorflow as tf
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.python.framework import ops
import math
import matplotlib.pyplot as plt

DATA = "../dir/cs341-driver-data/nervtech/v1/drives-with-collisions/user_1636_scenario_0_repeat_0_opti.csv"


def prep_data(file=DATA, label='SPEED_LIMIT', split=0.9):
    df = pd.read_csv(file)
    df = df.drop(['DATE'], axis=1)
    print('DF SHAPE: {}'.format(df.shape))
    # Encoding class labels
    class_to_predict = label
    class_mapping = {label:idx for idx,label in enumerate(np.unique(df[class_to_predict]))}
    print(class_mapping)

    y = []
    for i, row in df.iterrows():
        lbl = row[class_to_predict]
        index = class_mapping[lbl]
        zeros = np.zeros(len(class_mapping.keys()))
        zeros[index] = 1.0
        y.append(zeros)

    df[class_to_predict] = df[class_to_predict].map(class_mapping)

    X = df.drop([class_to_predict], axis=1).as_matrix()
    y = np.asarray(y)
    print(y[:10])


    X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = split)

   
    return X_train.T, X_test.T, Y_train.T, Y_test.T

def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
    return X, Y

def initialize_parameters():
    W1 = tf.get_variable("W1", [50, 117], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [50,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [25,50], initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [25,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [12,25], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [12,1], initializer = tf.zeros_initializer())
    W4 = tf.get_variable("W4", [7,12], initializer = tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", [7,1], initializer = tf.zeros_initializer())
    parameters = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3,
              "W4": W4,
              "b4": b4}

    return parameters

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']


    Z1 = tf.add(tf.matmul(W1, X), b1)                                             
    A1 = tf.nn.relu(Z1)                                          
    Z2 = tf.add(tf.matmul(W2, A1), b2)                                            
    A2 = tf.nn.relu(Z2)                                              
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    A3 = tf.nn.softmax(Z3)
    Z4 = tf.add(tf.matmul(W4, A3), b4)                                             

    return Z4

def compute_cost(Z4, Y):
    print("Shape of Z4: {}".format(Z4.shape))
    logits = tf.transpose(Z4)
    print("LOGITS SHAPE: {}".format(logits.get_shape()))
    print("---Printing Logits---")

    # print_logits = tf.Print(logits, [logits])

    print("Shape of Y: {}".format(Y.shape))
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    return cost

def get_mini_batches(X, Y, mini_batch_size = 64):
    m = X.shape[0]                
    mini_batches = []

    print(m)

    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1000, print_cost = True):

    ops.reset_default_graph()
    n_x, m = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z4 = forward_propagation(X, parameters)
    cost = compute_cost(Z4, Y)
    optimizer =  tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    minibatching = False

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.                      
            
            if minibatching: 
                minibatches = get_mini_batches(X_train, Y_train, minibatch_size)
                for minibatch in minibatches:
                    minibatch_X, minibatch_Y = minibatch
                    _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                    epoch_cost += minibatch_cost / num_minibatches
            else:
                _ , batch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
                epoch_cost += batch_cost / m
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)

        correct_prediction = tf.equal(tf.argmax(Z4), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


X_train, X_test, Y_train, Y_test = prep_data()
parameters = model(X_train, Y_train, X_test, Y_test)

