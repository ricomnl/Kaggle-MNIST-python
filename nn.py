import numpy as np
import math
import matplotlib.pyplot as plt
from forward import forward_propagation
from backward import backward_propagation
from gradcheck import gradient_check

def one_hot_matrix(labels, C):
    one_hot = np.zeros((labels.shape[0], C))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot

def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims) - 1 #number of layers
    
    for l in range(1, L + 1):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt([2/layer_dims[l-1]])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def random_mini_batches(X, Y, mini_batch_size, c):
    m = X.shape[1]
    mini_batches = []
    
    #Step 1: Shuffle
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((c,m))
    
    #Step 2: Partition
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    #Handling the end case
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, : m - mini_batch_size * num_complete_minibatches]
        mini_batch_Y = shuffled_Y[:, : m - mini_batch_size * num_complete_minibatches]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    return mini_batches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = 0
    
    cost = np.sum(np.multiply(Y, np.log(AL)))
    cost = (-1/m) * cost    
    cost = np.squeeze(cost)
    assert(cost.shape ==())
    
    return cost

def compute_cost_with_regularization(AL, Y, parameters, lambd):
    m = Y.shape[1]
    L = len(parameters) // 2
    L2_regularization_cost = 0
    
    cross_entropy_cost = compute_cost(AL, Y)    

    for l in range(L):
        L2_regularization_cost += (lambd/(2*m)) * np.sum(np.square(parameters["W" + str(l+1)]))
        
    cost = cross_entropy_cost + L2_regularization_cost
    return cost
        

def update_parameters_with_gd(parameters, grads, learning_rate):
    L = len(parameters) // 2
    
    for l in range(L):
        parameters["W" + str(l+1)] += - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] += - learning_rate * grads["db" + str(l+1)]
        
    return parameters

def L_layer_model(X, Y, layer_dims, optimizer, learning_rate = 0.0075, mini_batch_size = 64, 
                  beta = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_epochs = 3000, 
                  print_cost=True, lambd = 0, keep_prob = 1, gradcheck = 0):
    grads = {}
    costs =  []
    t = 0 #initializing the counter for Adam 
    m = X.shape[1]
    difference = 0 #for gradient check
    
    #Initialize parameters dictionary
    parameters = initialize_parameters(layer_dims)
    
    #Initialize the optimizer
    if optimizer == "gd":
        pass
    #elif optimizer == "adam": 
    #    v,s = initialize_adam(parameters)
    
    #Optimization loop
    for i in range(0, num_epochs):
        
        minibatches = random_mini_batches(X, Y, mini_batch_size, Y.shape[0])
        
        for minibatch in minibatches:
            
            #Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            
            #Forward propagation
            if keep_prob == 1:
                AL, caches = forward_propagation(minibatch_X, parameters)
            #elif keep_prob < 1:
            #    AL, caches = forward_propagation_with_dropout(X, parameters, keep_prob)
            
            #Compute cost
            if lambd == 0:
                cost = compute_cost(AL, minibatch_Y)
            else:
                cost = compute_cost_with_regularization(AL, minibatch_Y, parameters, lambd)
            
            #Backward propagation
            grads = backward_propagation(AL, minibatch_Y, caches, lambd)
            
            #Gradient check
            if gradcheck == 1 and i == 0:
                print("Gradient check ...")
                difference = gradient_check(parameters, grads, minibatch_X, minibatch_Y)
            
            #Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            #elif optimizer == "adam":
            #    t += 1
            #    parameters, v, s= update_paramters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
        
        #Print cost every 100 training example
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

