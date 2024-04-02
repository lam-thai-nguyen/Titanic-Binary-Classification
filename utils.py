import copy
import math
import numpy as np
import matplotlib.pyplot as plt


def _sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """
    g = 1 / (1 + np.exp(-z))

    return g


def _compute_cost_logistic(X, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """

    m = X.shape[0]
    cost = 0.0
    for i in range(m):  # Loop through each example
        y_hat = _sigmoid(np.dot(w, X[i]) + b)
        loss = -y[i] * np.log(y_hat) - (1 - y[i]) * np.log(1 - y_hat)
        cost += loss

    cost /= m

    return cost


def _compute_gradient_logistic(X, y, w, b):
    """
    Computes the gradient for logistic regression

    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape
    dJ_dw = np.zeros((n,))
    dJ_db = 0.0

    for i in range(m):
        y_hat = _sigmoid(np.dot(w, X[i]) + b)
        e = y_hat - y[i]
        for j in range(n):
            dJ_dw[j] += e * X[i, j]
        dJ_db += e

    dJ_dw /= m
    dJ_db /= m

    return dJ_dw, dJ_db


def gradient_descent(X, y, w_in, b_in, alpha, num_iters, plot_cost_per_iter):
    """
    Performs batch gradient descent

    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent
      plot_cost_per_iter (bool)

    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter
    """
    w = copy.deepcopy(w_in)
    b = b_in
    cost_record = []

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_dw, dj_db = _compute_gradient_logistic(X, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        cost = _compute_cost_logistic(X, y, w, b)
        cost_record.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"{i}th cost: {cost}")
            
    if plot_cost_per_iter:
        _plot_cost_iter(cost_record)

    return w, b


def _plot_cost_iter(cost_record: list[float]) -> None:
    """
    Plot the relationship between cost value and number of iterations
    """
    plt.plot(cost_record)
    plt.xlabel("Iter")
    plt.ylabel("Cost")
    plt.show()
    
    
def predict(X_new, w, b):
    """
    Args:
        X_new (ndarray): new example
        w (ndarray): model weights
        b (float): model bias
    Returns:
        y_hat
    """
    y_hat = _sigmoid(np.dot(w, X_new) + b)
    if y_hat >= 0.5:
        y_hat = 1
    else:
        y_hat = 0
        
    return y_hat
