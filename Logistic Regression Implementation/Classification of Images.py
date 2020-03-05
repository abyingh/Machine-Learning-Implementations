"""
                                  
                                  Classification of '0' and '1' in sign language
                                  
Data can be downloaded from:

                            https://www.kaggle.com/ardamavi/sign-language-digits-dataset

"""



import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


x = np.load(path).reshape(2062, -1)
y = np.load(path)
y = np.array([np.argmax(i) for i in y]).reshape(-1,1)

idx = np.where(y<=1)
y = y[idx].reshape(-1,1)
x = x[idx[0]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 13)




def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_and_gradients(w, b, x, y):
    z = np.dot(x,w) + b
    y_head = sigmoid(z)

    J = np.mean(-y * np.log(y_head) - (1-y) * np.log(1 - y_head))
    db = (y_head-y)/x.shape[0]
    dw = (np.dot(x.T, (y_head-y)))/x.shape[0]

    return J, dw, db

def update_parameters(w, b, dw, db, lr):
    w -= lr * dw
    b -= lr * db

    return w, b

def logistic_regression(x, y, lr, x_test, y_test):
    costs = []
    iterations = []
    
    w = np.zeros((x.shape[1], 1), dtype=np.float)
    b = np.full((x.shape[0], 1), -1, dtype=np.float)

    prev_cost = 0

    i = 1
    while True:
        J, dw, db = cost_and_gradients(w, b, x, y)
        if abs(J - prev_cost) < 1e-5:
            break
        prev_cost = J

        w, b = update_parameters(w, b, dw, db, lr)

        i += 1
        if i % 10 == 0:
            costs.append(J)
            iterations.append(i)

    plt.plot(iterations, costs)

    y_pred = sigmoid(np.dot(x, w) + b)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in y_pred]).reshape(-1,1)

    y_pred_test = sigmoid(np.dot(x_test, w) + b[:x_test.shape[0]])
    y_pred_test = np.array([1 if i >= 0.5 else 0 for i in y_pred_test]).reshape(-1,1)


    r2_train = r2_score(y, y_pred)
    r2_test = r2_score(y_test, y_pred_test)

    return r2_train, r2_test

logistic_regression(x_train, y_train, 0.01, x_test, y_test)
