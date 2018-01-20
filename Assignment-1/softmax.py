import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg) :
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train) :
        scores = X[i].dot(W)
        scores -= np.max(scores)  # prevents numerical instability
        correct_class_score = scores[y[i]]

        exp_sum = np.sum(np.exp(scores))
        loss += np.log(exp_sum) - correct_class_score

        dW[:, y[i]] -= X[i]
        for j in range(num_classes) :
            dW[:, j] += (np.exp(scores[j]) / exp_sum) * X[i]

    loss/=num_train
    loss+=0.5*reg*np.sum(W*W)
    dW/=num_train
    dW+=reg*W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg) :
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train, dim = X.shape

    f = X.dot(W)  # (N,C)

    f_max = np.reshape(np.max(f, axis=1), (num_train, 1))  # (N,1)
    prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True)
    y_trueClass = np.zeros_like(prob)
    y_trueClass[range(num_train), y] = 1.0  # N by C
    loss += -np.sum(y_trueClass * np.log(prob)) / num_train + 0.5 * reg * np.sum(W * W)
    dW += -np.dot(X.T, y_trueClass - prob) / num_train + reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
