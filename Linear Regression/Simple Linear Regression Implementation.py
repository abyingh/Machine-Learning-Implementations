import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def mse(x_train, y_train, w0, w1):
    y_pred = w0 + w1 * x_train
    residuals = y_pred - y_train
    
    return np.mean(np.sum(np.square(residual)), axis= 1)
    
    
def gradients(x_train, y_train, w0, w1):
    y_pred = w0 + w1 * x_train
    residuals = y_pred - y_train
    
    return 2 * np.sum(residuals), 2 * np.sum(np.multiply(residuals * x_train))


def simple_linear_regression(x_train, y_train, learning_rate, eps, x_test, y_test):
    n = x_train.shape[0]
    m = x_test.shape[0]
    w0 = np.zeros((n,1))
    w1 = np.zeros((n,1))
    
    i = 1
    previous_cost = 0
    
    while True:
        cost = mse(x_train, y_train, w0, w1)
        
        if np.abs(previous_cost - cost) < eps:
            break
            
        previous_cost = cost  
        
        grad1, grad2 = gradients(x_train, y_train, w0, w1)
        
        w0 -= learning_rate * grad1
        w1 -= learning_rate * grad2
        
        if i % 1000 == 0:
            print('Mean Square Error = {}'.format(mse))
    
    y_test_pred = w0[m] + w1[m] * x_test
    
    return y_test_pred, r2_score(y_test,y_test_pred)
