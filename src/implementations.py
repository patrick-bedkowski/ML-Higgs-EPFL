import numpy as np


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):

    def loss_function_SGD(y, tx, w):
        sample_index = np.random.randint(0, len(y))
        ySGD = y[sample_index]
        xSGD = np.array([tx[sample_index, :]])
        return compute_loss_and_gradient_least_squares(ySGD, xSGD, w)

    def compute_loss_and_gradient_least_squares(y, tx, w):
        """
        :return: Loss, gradient
        """

        n = len(tx)
        error = y - tx.dot(w)  # y - tx @ w
        gradient = -1/n * tx.T.dot(error)  # tx.T @ error
        loss = 1/(2*n) * error.T.dot(error)
        return loss, gradient

    return gradient_descent(loss_function_SGD, initial_w, max_iters, gamma, y, tx)


def logistic_regression(y, tx, initial_w, max_iters, gamma):

    def logistic_regression_loss_function(y, tx, w):
        """
        Least Squares loss function.
        :param y: target
        :param tx: data
        :param w: weights
        :return: loss, gradient
        """
        y_pred = sigmoid(tx.dot(w))
        epsilon = 1e-7  # to make logloss stable log(0) complains

        # compute the function value
        # f = 1/len(y)*np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))
        f = np.sum(-y * np.log(y_pred + epsilon) - (1 - y) * np.log(1 - y_pred + epsilon))  # scalar

        # compute the gradient value
        g = tx.T @ (y_pred-y)

        return f, g

    def loss_function_sgd(y, tx, w):
        # select a single example to compute the loss and the gradient
        index = np.random.randint(0, len(y))
        ySGD = y[index]
        xSGD = np.array([tx[index,:]])
        return logistic_regression_loss_function(ySGD, xSGD, w)

    return gradient_descent(loss_function_sgd, initial_w, max_iters, gamma, y, tx)


def gradient_descent(loss_function, initial_w, max_iters, gamma, y, tx):
    w = initial_w
    iter_num = 0
    errors = []
    while iter_num < max_iters:
        error, gradient_temp = loss_function(y, tx, w)
        w_new = w - gamma * gradient_temp
        w = w_new

        errors.append(error)
        iter_num += 1
    return w, errors


def sigmoid(x):
    return 1/(1+np.e**(-x))
