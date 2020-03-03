import numpy as np
from sklearn.metrics import r2_score


def mse(x_train, y_train, w):
    y_pred = np.matmul(x_train, w)
    return np.mean(np.square(y_pred - y_train))
    
    
def gradient(x_train, y_train, w):
    y_pred = np.matmul(x_train, w)
    return 2*np.mean(np.matmul(x_train.T, (y_pred - y_train) ), axis= 1)


def multiple_linear_regression(x_train, y_train, learning_rate, eps, x_test, y_test):
    n = x_train.shape[1]
    w = np.zeros((n,1))
    
    i = 1
    previous_cost = 0
    
    while True:
        cost = mse(x_train, y_train, w)
        
        if np.abs(previous_cost - cost) < eps:
            break
            
        previous_cost = cost  
        
        grad = gradients(x_train, y_train, w0, w1)
        
        w -= learning_rate * grad
        
        if i % 1000 == 0:
            print('Mean Square Error = {}'.format(mse))
    
    y_test_pred = np.matmul(x_test, w)
    
    return y_test_pred, r2_score(y_test,y_test_pred)
