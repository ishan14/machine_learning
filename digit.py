import pandas as pd
import numpy as np

X_train = pd.read_csv('C:/Users/ishan/data/digit-recognizer/train.csv') # loding normalized dataset 
X_test = pd.read_csv('C:/Users/ishan/data/digit-recognizer/test.csv')

#Y_train = X_train.iloc[:,0]
X_train_orig = X_train.to_numpy()
X_test_orig  = X_test.to_numpy()
#Y_train_orig = Y_train.to_numpy()

X = X_train_orig[:,1:]
Y = X_train_orig[:,0:1]

m = X.shape[0]
layer_dims = [784,28,14,10] # 4-Layer neural network with the following number of nodes for each layer.

## INPUT->RELU->RELU->SOFTMAX

Y_one_hot = np.zeros((m,10))                #  making one_hot_matrix for # 
                                            #  multi-class               #
for i in range(m):                          #  classification            #
    j = int(Y[i])                           #                            #
    Y_one_hot[i][j] = Y_one_hot[i][j] + 1   #                            #
print(Y_one_hot.shape)                      #                            #

def initialize_parameters(layer_dims):
    #Layer1 = 784
    #Layer2 = 28
    #Layer3 = 14
    #Layer4 = 10
    parameters = {}
    L = len(layer_dims)
    np.random.seed(1)
    
    for l in range(1,L):
        parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01
        parameters['b'+str(l)] = np.zeros((layer_dims[l],1))
    
    return parameters
    
def reLu(matrix):
    return np.maximum(0,matrix)

def softmax(matrix):
    max = np.max(matrix)
    return np.exp(matrix-max)/sum(np.exp(matrix-max))

def relu_derivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def forward_propogation(parameters,X):
    
    cache = {}
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = W1@X.T + b1
    A1 = reLu(Z1)
    Z2 = W2@A1 + b2
    A2 = reLu(Z2)
    Z3 = W3@A2 + b3
    A3 = softmax(Z3)
    
    cache = {'Z1':Z1,'A1':A1,'Z2':Z2,'A2':A2,'Z3':Z3,'A3':A3}
    
    return cache

def compute_cost(cache,X,Y):
    
    A3 = cache['A3']
    m = X.shape[0]
    
    #cost = (-1/m) * np.sum(np.multiply(Y,np.log(A3)),np.multiply((1-Y),np.log(1-A3))) 
    cost = -np.mean(Y*np.log(A3.T + np.exp(-8)))
    return cost


def backward_propogation(cache,parameters,Y):
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = cache['Z1']
    A1 = cache['A1']
    Z2 = cache['Z2']
    A2 = cache['A2']
    Z3 = cache['Z3']
    A3 = cache['A3']
    
    gradients = {}
    
    dZ3 = A3 - Y.T
    dW3 = (1/m)* dZ3@A2.T
    db3 = (1/m) * np.sum(dZ3,axis=1,keepdims=True)
    dZ2 = W3.T@dZ3 * relu_derivative(Z2)
    dW2 = (1/m)* dZ2@A1.T
    db2 = (1/m) * np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = W2.T@dZ2 * relu_derivative(Z1)
    dW1 = (1/m)* dZ1@X
    db1 = (1/m) * np.sum(dZ1,axis=1,keepdims=True)
    
    gradients = {'dZ3':dZ3,'dW3':dW3,'db3':db3,
                 'dZ2':dZ2,'dW2':dW2,'db2':db2,
                 'dZ1':dZ1,'dW1':dW1,'db1':db1}
    
    return gradients

def optimize(layers_dims,parameters,gradients,learning_rate):
    
    L = len(layer_dims)
    for l in range(1,L):
        parameters['W'+str(l)] = parameters['W'+str(l)] - learning_rate * gradients['dW'+str(l)]
        parameters['b'+str(l)] = parameters['b'+str(l)] - learning_rate * gradients['db'+str(l)]
        
    return parameters


def model(X,Y,layer_dims,learning_rate = 0.0075,num_iterations = 2000,print_cost=False):
    
    np.random.seed(1)
    grads = {}
    costs = []
    m = X.shape[0]
    
    parameters = initialize_parameters(layer_dims)
    
    for i in range(0,num_iterations):
        
        cache = forward_propogation(parameters,X)
        
        cost = compute_cost(cache,X,Y)
        
        gradients = backward_propogation(cache,parameters,Y)
        
        parameters = optimize(layer_dims,parameters,gradients,learning_rate)
        
        if print_cost and i%100 == 0:
            print('Cost after iteration {}: {}'.format(i,cost))
            costs.append(cost)
    
    
    return parameters    

parameters = model(X,Y_one_hot,layer_dims=[784,28,14,10],learning_rate=0.075,num_iterations=2000,print_cost=True)