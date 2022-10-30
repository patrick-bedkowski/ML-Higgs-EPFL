import numpy as np

def compute_loss(y, tx, w, type='mse'):

    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.
        type: string that can take value 'mse' or 'mae'.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # Compute loss by MSE
    error = y - np.dot(tx, w)
    N = len(tx)
    if type == 'mse':
        loss = 1/(2*N)*np.sum(np.square(error))
    elif type == 'mae':
        loss = 1/N*np.sum(np.abs(error))

    return loss

""" ------------------------ Least squares ------------------------ """

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
        
    """
    # Returns mse, and optimal weights
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    mse = compute_loss(y, tx, w, 'mse')
    return w, mse

""" ------------------------ Regularized logistic regression ------------------------ """

def calculate_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 

    Returns:
        a non-negative loss

    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    N = len(y)
    first_component = np.log(1+np.exp(tx.dot(w)))
    second_component = y*(tx.dot(w))
    cost = sum(first_component-second_component)/N

    return float(cost)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def calculate_logistic_loss_gradient(y, tx, w):
    """Compute the gradient of logistic loss.
    
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 

    Returns:
        a vector of shape (D, 1)

    """
    N = len(y)
    grad = 1/N*tx.T.dot(sigmoid(tx.dot(w))-y)
    return grad

def penalized_logistic_regression(y, tx, w, lambda_):
    """Return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)

    """

    loss = calculate_logistic_loss(y, tx, w) + lambda_*w.T@w
    gradient = calculate_logistic_loss_gradient(y, tx, w) + 2*lambda_*w

    return float(loss), gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar
        lambda_: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma*gradient
    return loss, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """The regularized logistic regression gradien descent (GD) algorithm.

    Args:
        y:          shape=(N, 1)
        tx:         shape=(N, D)
        lambda_:    scalar
        initial_w:  shape=(D, 1)
        max_iters:  scalar
        gamma:      scalar
        

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    """
    # init parameters
    w = initial_w
    losses = []
    threshold = 1e-8
    

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss


def mean_squared_error_SGD(y, tx, initial_w, max_iters, gamma):

    def loss_function_SGD(y, tx, w):
        sample_index = np.random.randint(0, len(y))
        ySGD = y[sample_index]
        xSGD = np.array([tx[sample_index, :]])
        return compute_loss_and_gradient_mean_squared(ySGD, xSGD, w)

    def compute_loss_and_gradient_mean_squared(y, tx, w):
        """
        :return: Loss, gradient
        """

        n = len(tx)
        error = y - tx.dot(w)  # y - tx @ w
        gradient = -1/n * tx.T.dot(error)  # tx.T @ error
        loss = 1/(2*n) * error.T.dot(error)
        return loss, gradient

    return gradient_descent(loss_function_SGD, initial_w, max_iters, gamma, y, tx)


def mean_squared_error_GD(y, tx, initial_w, max_iters, gamma):

    def loss_function_GD(y, tx, w):
        return compute_loss(y, tx, w, type='mse')

    return gradient_descent(loss_function_GD, initial_w, max_iters, gamma, y, tx)


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


def ridge_regression(y, tx, lambda_):
    LAMBDA = 2 * len(y) * lambda_

    # Solve Normal Equations with Regularization Term
    w = np.linalg.solve((tx.T @ tx + (np.eye(len(tx[1])) * LAMBDA)), tx.T @ y)

    return w


# Compute the gradient
def compute_gradient(y, tx, w):
    err = y - tx.dot(w)
    grad = - tx.T.dot(err) / len(err)

    return grad


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w, type='mse')

        w = w - gamma * grad

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
