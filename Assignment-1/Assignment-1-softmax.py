import random
import time
import numpy as np
import matplotlib.pyplot as plt
from gradient_check import grad_check_sparse
from CIFAR import load_CIFAR10
from softmax import *
from linear_classifier import *
def get_CIFAR10_data(num_training=49000,num_validation=1000,num_test=1000,num_dev=500):

    # Load the raw CIFAR-10 data
    cifar10_dir='E:/research/CS231n/cifar-10-python/cifar-10-batches-py'
    X_train,y_train,X_test,y_test=load_CIFAR10(cifar10_dir)

    # sub sample the data
    mask=list(range(num_training,num_training+num_validation))
    X_val=X_train[mask]
    y_val=y_train[mask]
    mask=list(range(num_training))
    X_train=X_train[mask]
    y_train=y_train[mask]
    mask=list(range(num_test))
    X_test=X_test[mask]
    y_test=y_test[mask]
    mask=np.random.choice(num_training,num_dev,replace=False)
    X_dev =X_train[mask]
    y_dev=y_train[mask]

    #Preprocessing : reshape the image data into rows
    X_train=np.reshape(X_train,(X_train.shape[0],-1))
    X_val=np.reshape(X_val,(X_val.shape[0],-1))
    X_test=np.reshape(X_test,(X_test.shape[0],-1))
    X_dev=np.reshape(X_dev,(X_dev.shape[0],-1))

    # Normalize the data: subtract the mean image
    mean_image=np.mean(X_train,axis=0)
    X_train-=mean_image
    X_val-=mean_image
    X_test-=mean_image
    X_dev-=mean_image

    #abb bias dimension and transform in to columns
    X_train=np.hstack([X_train,np.ones((X_train.shape[0],1))])
    X_val=np.hstack([X_val,np.ones((X_val.shape[0],1))])
    X_test=np.hstack([X_test,np.ones((X_test.shape[0],1))])
    X_dev=np.hstack([X_dev,np.ones((X_dev.shape[0],1))])

    return X_train,y_train,X_val,y_val,X_test,y_test,X_dev,y_dev






plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# Invoke the above function to get our data.
X_train ,y_train,X_val,y_val,X_test,y_test,X_dev,y_dev=get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
print('dev data shape: ', X_dev.shape)
print('dev labels shape: ', y_dev.shape)

# ## softmax Classifier
#
# Your code will all be written inside "softmax.py"

# Generate a random softmax weight matrix and use it to compute hte loss.
W=np.random.randn(3073,10) *0.0001
loss,grad=softmax_loss_naive(W,X_dev,y_dev,0.0)

# As a rough sanity check, our loss should be something close to -log(0.1).
print('loss: %f' % loss)
print('sanity check: %f' % (-np.log(0.1)))

# Complete the implementation of softmax loss naive.
loss,grad=softmax_loss_naive(W,X_dev,y_dev,0.0 )

# As we did for the SVM, use numeric gradient checking as a debugging tool.
# The numeric gradient should be close to the analytic gradient.
f=lambda W: softmax_loss_naive(W,X_dev,y_dev,0.0)[0]
grad_numerical=grad_check_sparse(f,W,grad,10)

# similar to SVM case, do another gradient check with regularization
loss,grad=softmax_loss_naive(W,X_dev,y_dev,5e1)
f=lambda W: softmax_loss_naive(W,X_dev,y_dev,5e1)[0]
grad_numerical=grad_check_sparse(f,W,grad,10)


# Implement a vectorized version in softmax loss vectorized.
tic=time.time()
loss_naive,grad_naive=softmax_loss_naive(W,X_dev,y_dev,0.000005)
toc=time.time()
print('naive loss: %e computed in %fs' % (loss_naive,toc-tic))

tic=time.time()
loss_vectorized,grad_vectorized=softmax_loss_vectorized(W,X_dev,y_dev,0.000005)
toc=time.time()
print('vectorized loss: %e computed in %fs' % (loss_vectorized,toc-tic))

# As we did for the SVM, we use the Frobenius norm to compare the two versions
# of the gradient.
grad_difference=np.linalg.norm(grad_naive-grad_vectorized,ord='fro')
print('Loss difference: %f' % np.abs(loss_naive-loss_vectorized))
print('Gradient difference: %f' % grad_difference)


# Use the validation set to tunehyperparameters

results={}
best_val=-1
best_softmax=None
learning_rates=[1e-7,5e-7]
regularization_strengths=[2.5e4,5e4]

# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained softmax classifer in best_softmax.                          #
params=[(x,y) for x in learning_rates for y in regularization_strengths]
for lrate, regular in params:
    softmax=Softmax()
    loss_hist=softmax.train(X_train,y_train,learning_rate=lrate,reg=regular,num_iters=700,verbose=True)
    y_train_pred=softmax.predict(X_train)
    accuracy_train=np.mean(y_train==y_train_pred)
    y_val_pred=softmax.predict(X_val)
    accuracy_val=np.mean(y_val==y_val_pred)
    results[(lrate,regular)]=(accuracy_train,accuracy_val)
    if(best_val<accuracy_val):
        best_val=accuracy_val
        best_softmax=softmax

for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print ('lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy))
    print ('best validation accuracy achieved during cross-validation: %f' % best_val)
