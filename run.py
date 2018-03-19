from nn import L_layer_model, one_hot_matrix
import numpy as np
import pandas as pd
from nn import initialize_parameters
from gradcheck import dictionary_to_vector,vector_to_dictionary
from predict import predict, get_error
from sklearn.cross_validation import train_test_split

dataset = pd.read_csv("train.csv", sep = ",", header = 0)
testset = pd.read_csv("test.csv", sep = ",", header = 0)
X = dataset.iloc[:, 1:]
Y = dataset.iloc[:, 0]
m = X.shape[0]

#X_cv = X.iloc[m-10000:, :].values
#Y_cv = Y.iloc[m-10000:].values
#X = X.iloc[ :m-10000, :].values
#Y = Y.iloc[ :m-10000].values
X, X_cv, Y, Y_cv = train_test_split(X.values, Y.values, test_size = 0.25)

Y_cv = one_hot_matrix(Y_cv, C = 10)
Y = one_hot_matrix(Y, C = 10)

#CONSTANTS
layer_dims = [784, 150, 150, 150, 100, 100, 10]
#layer_dimsG = [784, 10, 10]
#alpha = [0.0001, 0.0003, 0.003, 0.001]
#lambd = [1, 2, 5, 10, 15]
#lambd = [26, 27, 28, 29]
#keep_prob = [0.86, 0.89, 0.91, 0.93, 0.95]

#parameters2 = L_layer_model(X[0:1000, :].T, Y[0:1000, :].T, layer_dimsG, optimizer = "gd", learning_rate = 0.001, mini_batch_size = 1000,
#              beta = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_epochs = 100,
#              print_cost = True, lambd = 0, keep_prob = 1, gradcheck = 1)

#for i in range(len(lambd)):    
#print("alpha = " + str(lambd[i]))
parameters = L_layer_model(X.T, Y.T, layer_dims, optimizer = "adam", learning_rate = 0.0004, mini_batch_size = 1024, 
          beta = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_epochs = 1000, 
          print_cost=True, lambd = 26, keep_prob = 0.85, gradcheck = 0)

Yhat = predict(X.T, parameters)
error = get_error(Yhat, Y)
Yhat_cv = predict(X_cv.T, parameters)
error_cv = get_error(Yhat_cv, Y_cv)

print("Training set error: %f" %(error))
print("Cross validation set error: %f" %(error_cv))
print()
print()

X_test = testset.values
pred = predict(X_test.T, parameters)
np.savetxt("./Submissions/submission.csv", pred, delimiter =";", fmt="%i")