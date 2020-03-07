"""
Data is the same as in the logistic regression. 

--- SHAPES for initialization ---

x =  (samples, pixels)
w1 = (pixels, neurons)
b1 = (1, neurons)
--> a1 = (samples, neurons) 

w2 = (neurons, classes)
b2 = (1, classes)
--> a2 = (samples, classes)

"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

x = np.load('/home/bora/Desktop/X.npy').reshape(2062, -1)
y = np.load('/home/bora/Desktop/Y.npy')
y = np.array([np.argmax(i) for i in y]).reshape(-1,1)
idx = np.where(y<=1)
y = y[idx].reshape(-1,1)
x = x[idx[0]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 13)


# def tanh(z):
#     return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

# def relu(z):
#     return max(0.1*z, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def tanh_derivative(z):
    return 1 - np.square(np.tanh(z))

def predict(x,w1,b1,w2,b2):
    z1 = np.dot(x, w1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    
    return a2

def cost_and_gradients(x, y, w1, b1, w2, b2):

    z1 = np.dot(x, w1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2) # y^

    # Cross entropy:
    J = - np.mean(np.multiply(np.log(a2), y))

    dz2 = (y - a2) # * sigmoid_derivative(z2)             # (327, 1)
    db2 = np.sum(dz2) / x.shape[0]                        # (1, 1)
    dw2 = np.dot(a1.T, dz2) / x.shape[0]                  # (10, 1)
    dz1 = np.dot(dz2, w2.T) * (1 - np.square(a1))         # (327, 10)
    db1 = np.sum(dz1 / x.shape[0], axis= 0) / x.shape[0]  # (1, 10)
    dw1 = np.dot(x.T, dz1)  / x.shape[0]                  # (4096, 10)

    return J, dw1, db1, dw2, db2, a2

def update(w1, b1, w2, b2, dw1, db1, dw2, db2, lr):
    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2

    return w1, b1, w2, b2

def two_layer_network(x, y, x_test, y_test, hidden_neurons, lr, epsilon):
    iterations = []
    costs = []
    parameters = dict(w1=[], b1=[], w2=[], b2=[])

    w1 = np.random.randn(x.shape[1], hidden_neurons) * 0.1
    w2 = np.random.randn(hidden_neurons, 1) * 0.1
    b1 = np.zeros((1, hidden_neurons), dtype=np.float)
    b2 = np.zeros((1, y.shape[1]), dtype=np.float)

    previous_cost = 0

    i = 1
    while True:
    for j in range(500):
        cost, dw1, db1, dw2, db2, a2 = cost_and_gradients(x, y, w1, b1, w2, b2)

        if np.abs(cost - previous_cost) < epsilon:
             break
        previous_cost = cost

        w1, b1, w2, b2 = update(w1, b1, w2, b2, dw1, db1, dw2, db2, lr)

        if i%100 == 0:
            costs.append(cost)
            iterations.append(i)
        i += 1

    plt.plot(iterations, costs, color='blue')

    y_pred = predict(x, w1, b1, w2, b2)
    y_pred_train = np.array([1 if i>= 0.5 else 0 for i in y_pred])

    r2_training = r2_score(y, y_pred)

    return r2_training

two_layer_network(x_train, y_train, x_test, y_test, 3, 0.01, 1e-1)
