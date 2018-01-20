import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
from gradient_check import grad_check_sparse
from random import shuffle
from linear_classifier import LinearSVM
from CIFAR import load_CIFAR10
# coding: utf-8

# # Multiclass Support Vector Machine exercise
#
# *Complete and hand in this completed worksheet (including its outputs and any supporting code outside of the worksheet) with your assignment submission. For more details see the [assignments page](http://vision.stanford.edu/teaching/cs231n/assignments.html) on the course website.*
#
# In this exercise you will:
#
# - implement a fully-vectorized **loss function** for the SVM
# - implement the fully-vectorized expression for its **analytic gradient**
# - **check your implementation** using numerical gradient
# - use a validation set to **tune the learning rate and regularization** strength
# - **optimize** the loss function with **SGD**
# - **visualize** the final learned weights
#




# SVM Classifier

def svm_loss_naive(W, X, y, reg) :
    """
    structured SVM Loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N exampoles.

    Inputs:
    :param W: A numpy array of shape(D,C) containing weights.
    :param X: A numpy array of shape(N,D) containig a minibatch of data.
    :param y: A numpy array of shape(N,) containig training labels; y[i] = c means
        that X[i] has label c , where 0 <= c < C.
    :param reg: (float) regularization strength
    :return: a tuple of :
        - loss as single float
        - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compoute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train) :
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes) :
            if j == y[i] :
                continue
            margin = scores[j] - correct_class_score + 1  # note delta=1
            if margin > 0 :
                loss += margin
                # calculate the dW, Sj - Syi + 1(j!=yi)
                dW[:, j] += X[i, :].T
                dW[:, y[i]] -= X[i, :].T

    # Right now the loss is a sum over lal training examples, but we want it
    # to be an average instead so we divide by num train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss
    loss += reg * np.sum(W * W)
    dW += reg*W

    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #




    return loss, dW


def svm_loss_vectorized(W, X, y, reg) :
    """
    parameters are the same as svm_loss_naive
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    scores=X.dot(W)     #Scores is a numpy array of shape(N,C) where each column is a score correspond to the class.
    num_train=X.shape[0]
    num_classes = W.shape[1]
    correct_class_scores = scores[range(num_train), list(y)].reshape(-1, 1)
    margins = np.maximum(0, scores - np.tile(correct_class_scores, (1, num_classes)) + 1)
    margins[range(num_train), list(y)] = 0

    loss = np.sum(margins)
    loss /= num_train

    loss += 0.5 * reg * np.sum(W * W)



    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    # keep only positive elements
    margins[margins > 0] = 1.0
    row_sum = np.sum(margins, axis=1)
    margins[np.arange(num_train), y] = -row_sum
    dW += np.dot(X.T, margins) / num_train + reg * W



    return loss, dW


plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load the raw CIFAR-10 data.
cifar10_dir = 'E:/research/CS231n/cifar-10-python/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# print out the size of data
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# num_classes = len(classes)
# samples_per_class = 7
# for y, cls in enumerate(classes):
#     idxs = np.flatnonzero(y_train == y)
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i * num_classes + y + 1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow(X_train[idx].astype('uint8'))
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls)
# plt.show()

# Split the data into train, val and test.
num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

# validation set will be num_validation points from the original training set.
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# training set will be the first num_train points from the original training set.
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# also make a development set, which is a small subset of the training set
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask]

# use the first num_test points of the original test set as our test set.
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# print out the dataset we are going to use.

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

# As a sanity check, print out the shapes of the data
print('Training data shape: ', X_train.shape)
print('Validation data shape: ', X_val.shape)
print('Test data shape: ', X_test.shape)
print('dev data shape: ', X_dev.shape)

# Preprocessing: subtract the mean image
# first: compute the image mean based on the training data
mean_image = np.mean(X_train, axis=0)
print(mean_image[:10])  # print a few of the elements
plt.figure(figsize=(4, 4))
plt.imshow(mean_image.reshape((32, 32, 3)).astype('uint8'))  # visualize the mean image
plt.show()

# second: subtract the mean image from train and test data
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
X_dev -= mean_image

# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)

# SVM Classifier

W = np.random.randn(3073, 10) * 0.0001

loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.000005)
print('loss: %f' % (loss,))

# The `grad` returned from the function above is right now all zero. Derive and implement the gradient for the SVM cost function and implement it inline inside the function `svm_loss_naive`. You will find it helpful to interleave your new code inside the existing function.
#
# To check that you have correctly implemented the gradient correctly, you can numerically estimate the gradient of the loss function and compare the numeric estimate to the gradient that you computed. We have provided code that does this for you:

# Once you've implemented the gradient, recompute it with the code below
# and gradient check it with the function we provided for you

loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.0)

# Numerically compute the gradient along several randomly chosen dimensions, and
# compare them with your analytically computed gradient. The numbers should match
# almost exactly along all dimensions.
f = lambda w : svm_loss_naive(w, X_dev, y_dev, 0.0)[0]
gradient_numerical = grad_check_sparse(f, W, grad)

# In

# Next implement the funcion svm_loss_vectorized;
tic = time.time()
loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Naive loss: %e computed in %fs' % (loss_naive, toc - tic))

tic = time.time()
loss_vectorized, grad_vectorized = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

# The losses should match but your vectorized implementation should be much faster
print('difference: %f' % (loss_naive - loss_vectorized))

# Complete the implementation of svm_loss_vectorized, and compute the gradient
# of the loss function in a vectorized way.

# THe naive implementation and the vectorized implementation should match, but
# the vectorized version should still be much faster.
tic = time.time()
_, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Naive loss and gradient: computed in %fs' % (toc - tic))

tic = time.time()
_, grad_vectorized = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Vectorized loss and gradient: computed in %fs' % (toc - tic))

# The loss is a single number, so it is easy to compare the values computed
# by the two implemetations. The gradient on the other hand is a matrix, so
# we use the Frobenius norm to compare them.
diffrence = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print('difference: %f' % diffrence)



# ### Stochastic Gradient Descent
#
# We now have vectorized and efficient expressions for the loss, the gradient and our gradient matches the numerical gradient. We are therefore ready to do SGD to minimize the loss.
svm=LinearSVM()
tic=time.time()
loss_hist=svm.train(X_train,y_train,learning_rate=1e-7 ,reg=2.5e4,num_iters=1500,verbose=True)

toc=time.time()
print('That took %fs ' % (toc-tic))


# A useful debugging strategy is to plot the loss as a function of
# iteration number:
plt.plot(loss_hist)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()

# Write the LinearSVM.predict function and evaluate the performance on both the
# training and validation set
y_train_pred=svm.predict(X_train)
print('training accuracy: %f ' % (np.mean(y_train == y_train_pred),))
y_val_pred=svm.predict(X_val)
print('validation accuracy: %f' % (np.mean(y_val == y_val_pred),))


# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of about 0.4 on the validation set.
learning_rates = [1e-7,5e-5]
regularization_strengths=[2.5e4,5e4]

# results is dictionary mapping tuples of the form
# (learning_rate, regularization_strength) to tuples of the form
# (training_accuracy, validation_accuracy). The accuracy is simply the fraction
# of data points that are correctly classified.
results={}
best_val = -1   # The highest validation accuracy that we have seen so far.
best_svm = None # The LinearSVM object that achieved the highest validation rate.

# Write code that chooses the best hyperparameters by tuning on the validation #
# set. For each combination of hyperparameters, train a linear SVM on the      #
# training set, compute its accuracy on the training and validation sets, and  #
# store these numbers in the results dictionary. In addition, store the best   #
# validation accuracy in best_val and the LinearSVM object that achieves this  #
# accuracy in best_svm.                                                        #
#                                                                              #
# Hint: You should use a small value for num_iters as you develop your         #
# validation code so that the SVMs don't take much time to train; once you are #
# confident that your validation code works, you should rerun the validation   #
# code with a larger value for num_iters.                                      #
params = [(x,y) for x in learning_rates for y in regularization_strengths]
for lrate, regular in params:
    svm = LinearSVM()
    loss_hist = svm.train(X_train, y_train, learning_rate=lrate, reg=regular,
                      num_iters=700, verbose=False)
    y_train_pred = svm.predict(X_train)
    accuracy_train = np.mean(y_train == y_train_pred)
    y_val_pred = svm.predict(X_val)
    accuracy_val = np.mean(y_val == y_val_pred)
    results[(lrate, regular)]=(accuracy_train, accuracy_val)
    if (best_val < accuracy_val):
        best_val = accuracy_val
        best_svm = svm

for lr, reg in sorted(results) :
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)

