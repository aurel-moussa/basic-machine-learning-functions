# Neural Network predicts using forward propagation processes
# This file makes use of a simple input layer (two nodes), one hidden layer (two nodes), one output layer (one node)

# import required packages
import numpy as np

weights = np.around(np.random.uniform(size=6), decimals=2) # initialize the weights randomly
biases = np.around(np.random.uniform(size=3), decimals=2) # initialize the biases randomly

#sanity check that everything was done correctly
print(weights)
print(biases)

#set input variables
x_1 = 0.5
x_2 = 0.85

print('x1 is {} and x2 is {}'.format(x_1, x_2))

#compute the weighted sum of first layer's first node 
z_1_1 = x_1 * weights[0] + x_2 * weights[1] + biases[0]
print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(z_1_1))

#compute the weighted sum of first layer's second node 
z_1_2 = x_1 * weights[2] + x_2 * weights[3] + biases[1]
print('The weighted sum of the inputs at the second node in the hidden layer is {}'.format(np.around(z_1_2, decimals=4)))

#we will be using a sigmoid activation function to compute the activation of the first layers's first node
a_1_1 = 1.0 / (1.0 + np.exp(-z_1_1))
print('The activation of the first node in the hidden layer is {}'.format(np.around(a_1_1, decimals=4)))

#and the same sigmoid activation function to compute the activation of the first layer's second node
a_1_2 = 1.0 / (1.0 + np.exp(-z_1_2))
print('The activation of the first node in the hidden layer is {}'.format(np.around(a_1_2, decimals=4)))

#these activation will be the inputs of the output layer
#let's compute that, shall we?

#compute the weighted sum of second layer's (output layer's) only node 
z_2 = a_1_1 * weights[4] + a_1_2 * weights[5] + biases[2]
print('The weighted sum of the inputs at the output node is {}'.format(np.around(z_2, decimals=4)))

#and let us now use our trusty friend, ol sigmoid, to compute the activation function
a_2 = 1.0 / (1.0 + np.exp(-z_2))
print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(np.around(a_2, decimals=4)))

