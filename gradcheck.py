import numpy as np
from forward import forward_propagation

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = 0
    
#    for j in range(1, Y.shape[0]):
#        cost += np.sum(Y[j, :] * np.log(AL[j, :]))
    cost = np.sum(np.multiply(Y, np.log(AL)))
    cost = (-1/m) * cost    
    #cost = (-1/m) * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))
    
    return cost

def dictionary_to_vector(parameters):
    L = len(parameters) // 2
    vec = np.zeros((0,0))
    
    for l in range(L):
        vec = np.append(vec, parameters["W" + str(l+1)].flatten()) 
        vec = np.append(vec, parameters["b" + str(l+1)].flatten())
    
    return vec

def gradients_to_vector(gradients):
    L = len(gradients) // 3
    vec = np.zeros((0,0))
    
    for l in range(L):
        vec = np.append(vec, gradients["dW" + str(l+1)].flatten()) 
        vec = np.append(vec, gradients["db" + str(l+1)].flatten())
    
    return vec

def vector_to_dictionary(theta, parameters):
    L = len(parameters) // 2
    dict = {}
    length = 0
    
    for l in range(L):
        dict["W" + str(l+1)] = np.array(theta[length : length + parameters["W" + str(l+1)].shape[0] * parameters["W" + str(l+1)].shape[1]].reshape(parameters["W" + str(l+1)].shape[0], parameters["W" + str(l+1)].shape[1]))
        length += parameters["W" + str(l+1)].shape[0] * parameters["W" + str(l+1)].shape[1]
        dict["b" + str(l+1)] = theta[length : length + parameters["b" + str(l+1)].shape[0]].reshape(parameters["b" + str(l+1)].shape[0], 1)
        length += parameters["b" + str(l+1)].shape[0]
        
    return dict
    

def gradient_check(parameters, gradients, X, Y, epsilon = 1e-7):
    parameters_values = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    for i in range(num_parameters):
        thetaplus = np.copy(parameters_values)
        thetaplus[i] = thetaplus[i] + epsilon
        AL, _ = forward_propagation(X, vector_to_dictionary(thetaplus, parameters))
        J_plus[i] = compute_cost(AL, Y)
        
        thetaminus = np.copy(parameters_values)
        thetaminus[i] = thetaminus[i] - epsilon
        AL, _ = forward_propagation(X, vector_to_dictionary(thetaminus, parameters))
        J_minus[i] = compute_cost(AL, Y)
        
        gradapprox[i] = (J_plus[i] - J_minus[i])/ (2*epsilon)
        
    numerator = np.linalg.norm(grad-gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator/denominator
    
    if difference > 2e-7:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference