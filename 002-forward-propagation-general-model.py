#Generalizable model for a forward propagation with some number of input layers, 
#some number of hidden layers, some number of nodes in the hidden layers, and some number of output layers

#import packages
import numpy as np #needed for randomization functions

#Defining structure of the model
n = 2 # number of inputs
num_hidden_layers = 2 # number of hidden layers
m = [2, 2] # number of nodes in each hidden layer
num_nodes_output = 1 # number of nodes in the output layer

#initialize the network by looping through all the layers and nodes and assigning random weights and biases

num_nodes_previous = n # number of nodes in the previous layer (starting with n, since the input layer is the first one)
network = {} # initialize network an an empty dictionary

# loop through each layer and randomly initialize weights and biases associated with each node
# adding 1 to the number of hidden layers in order to include the output layer as well
for layer in range(num_hidden_layers + 1): 
    
    # determine name of layer
    if layer == num_hidden_layers:
        layer_name = 'output'
        num_nodes = num_nodes_output #the output nodes
    else:
        layer_name = 'layer_{}'.format(layer + 1) #because python is 0-indexing, and we want thinks to be more human-comprehensible
        num_nodes = m[layer] #applying the correct number of nodes for each layer of the hidden parts
    
    # initialize weights and biases associated with each node in the current layer
    network[layer_name] = {}
    for node in range(num_nodes):
        node_name = 'node_{}'.format(node+1) #because python is 0-indexing, and we want thinks to be more human-comprehensible
        network[layer_name][node_name] = {
            'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
            'bias': np.around(np.random.uniform(size=1), decimals=2),
        }
    
    num_nodes_previous = num_nodes
    
print(network) # print network
