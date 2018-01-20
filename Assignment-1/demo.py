import numpy as np
import matplotlib.pyplot as plt

# a=np.array([[1,2,3],[4,5,6]])
# b=np.array([[3,2,1],[6,5,4]])
# print(a*b)
# print(np.dot(a,b.T))
# print(np.sum(a,axis=1))
# print(np.ones((5,1)))
# print(np.ones((5,1))*np.sum(a,axis=1))

# X_train=np.arange(40)
# num_folds=5
# X_train_folds=np.array_split(X_train,num_folds)
#
# for f in range(num_folds) :
#     X_train_tmp = np.array(X_train_folds[:f] + X_train_folds[f + 1 :])
#     print(X_train_tmp)

X=np.random.randn(2,5)
batch_inx = np.random.choice(2, 3)
X_batch = X[batch_inx,:]
print(X)
print(batch_inx)
print(X_batch)