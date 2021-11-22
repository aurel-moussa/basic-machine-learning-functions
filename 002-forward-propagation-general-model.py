#Generalizable model for a forward propagation with some number of input layers, 
#some number of hidden layers, some number of nodes in the hidden layers, and some number of output layers

#import packages
import numpy as np #needed for randomization functions

#Defining structure of the model
n = 2 # number of inputs
num_hidden_layers = 2 # number of hidden layers
m = [2, 2] # number of nodes in each hidden layer
num_nodes_output = 1 # number of nodes in the output layer

###### TRYING OUT THE LOOP ########
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

#### MAKING THE INITIZALIZATION OF THE NETWWORK INTO A NICE CLEAN FUNCTION ####
def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    
    num_nodes_previous = num_inputs # number of nodes in the previous layer
    network = {}
    
    # loop through each layer and randomly initialize weights and biases associated with each layer
    for layer in range(num_hidden_layers + 1): #include the output layer, too
        
        if layer == num_hidden_layers:
            layer_name = 'output' # name last layer in the network output
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1) # otherwise give the layer a number, and dont forget about 0-indexing
            num_nodes = num_nodes_hidden[layer] 
        
        # initialize weights and bias for each node
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1) #and dont forget about 0-indexing
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }
    
        num_nodes_previous = num_nodes

    return network # return the network

#The network initialization function has been defined, let us run this baby and create an aptly named small baby network!
small_baby_network = initialize_network(5, 3, [3, 2, 3], 1)

##### CREATING A FORWARD PROPAGATION ######
#we have initialized this network. 
#Put we do not yet know the weighted sum at each node. 
#For that we need to compute the dot product of inputs and weights, and then add the bias

def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

#let us seed a random input layer
from random import seed #yeah, yeah, i know, we already have it in the packages at the top, i just want to see it again
np.random.seed(12)
inputs = np.around(np.random.uniform(size=5), decimals=2)
print('The inputs to the network are {}'.format(inputs))

#just to see everything works, we will do a check run for the weighted sum computation of the layer 1 node 1
node_weights = small_baby_network['layer_1']['node_1']['weights']
node_bias = small_baby_network['layer_1']['node_1']['bias']
weighted_sum = compute_weighted_sum(inputs, node_weights, node_bias)
print('The weighted sum at the first node in the hidden layer is {}'.format(np.around(weighted_sum[0], decimals=4)))

#everything works. okay. next we need to define a function that does the activation using sigmoid
def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))

#just to check everything works, check run for the activation function of the layer 1 node 1
node_activation(weighted_sum)


####### PUTTING IT ALL TOGETHER FOR THE FORWARD PROPAGATION #######
#The final piece of building the neural network that can perform predictions is to put everything together 
#Create a function that applies the compute_weighted_sum and node_activation functions to each node in the network 
#and propagates the data all the way to the output layer and outputs a prediction for each node in the output layer

#1. Start with the input layer as the input to the first hidden layer
#2. Compute the weighted sum at the nodes of the current layer
#3. Compute the output (after activation function) of the nodes of the current layer
#4. Set the output of the current layer to be the input to the next layer
#5. Move to the next layer in the network
#6. Repeat steps 2 - 4 until we compute the output of the output layer

def forward_propagate(network, inputs):
    
    layer_inputs = list(inputs) # start with the input layer as the input to the first hidden layer
    
    for layer in network:
        
        layer_data = network[layer]
        layer_outputs = [] 
        for layer_node in layer_data:
        
            node_data = layer_data[layer_node]
        
            # compute the weighted sum and the output of each node at the same time 
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            #obvs this only works if you define the networks to have nice dictionary type with keys that are called weights and bias
            layer_outputs.append(np.around(node_output[0], decimals=4))
            
        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))
    
        layer_inputs = layer_outputs # set the output of this layer to be the input to next layer

    network_predictions = layer_outputs
    return network_predictions

#let us run a test on our small baby network with the inputs defined above
forward_propagate(small_baby_network, inputs)

#### PUTTING IT ALL TOGETHER - INITIALIZATION & FORWARD PROPAGATION ####
my_network = initialize_network(5, 3, [2, 3, 2], 3) #initialize network with a layer of x entry nodes, y layers of size z nodes each, and m output nodes
inputs = np.around(np.random.uniform(size=5), decimals=2) #get the inputs for the x entry nodes
predictions = forward_propagate(my_network, inputs) #and run our prediction
print('The predicted values by the network for the given input {} are {}'.format(inputs, predictions))
