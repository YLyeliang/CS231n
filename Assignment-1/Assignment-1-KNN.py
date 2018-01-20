import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from CIFAR import load_CIFAR10

class NearestNeighbor(object) :
    def __init__(self) :
        pass

    def train(self, X, y) :
        """X is N X D where each row is an example.Y is 1-dimension of size N"""
        # just remember all the data
        self.Xtr = X
        self.ytr = y

    def predict(self, X) :
        num_test = X.shape[0]
        # make sure the output type matches the input type
        Ypred = np.zeros(num_test, dtype=self.ytr.type)

        # loop over all test rows
        for i in range(num_test) :
            # find the nearest training image to the i'th test image(using L1distance)
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_index = np.argmin(distances)  # get the index with smallest distance
            Ypred[i] = self.ytr[min_index]  # predict the label of the nearest example

        return Ypred


# "其中有几个语句要注意一下:
# "X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
# "起初，X的size为(10000, 3072(3*32*32))。首先reshape很好理解，最后astype的格式转换也很好理解。
# "可是为什么要调用transpose，转置轴呢？就我认为只需要把一幅图像转成行向量就可以了。是为了方便检索吗？
# "xs.append(X)将5个batch整合起来；np.concatenate(xs)使得最终Xtr的尺寸为(50000,32,32,3)
# "当然还需要一步Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)使得每一副图像称为一个行向量，最终就有了50000个行向量（Xtr_rows的尺寸为（50000,3072））
# "综上，为了方便，难道不应该直接从最开始就不要调用reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")，直接append再concatenate不就能导出Xtr_rows了吗？"



class KNearestNeighbor(object) :
    """a Knn classifier with L2 distance"""

    def __init__(self) :
        pass

    def train(self, X, y) :
        """
        Train the classifier. Just memorizing the training data.
        Inputs:
        -X: A numpy array of shape(num_train,D) containing the training data
            consisting of num_train samples each of dimension D.
        -y: A numpy array of shape(N,) containing the training labels, where
            y[i] is the label for X[i]
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0) :
        """
        predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape(num_train,D) containing the training data
            consisting of num_train samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_Loops:O Determings which implementation to use to compute distances
            between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
            test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0 :
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1 :
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2 :
            dists = self.compute_distances_two_loops(X)
        else :
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X) :
        """
        Compute the distance between each test points in X and each training points
        in self.X_train using a nested loop over both the training data and the test data.

        Inputs:
        - X: A numpy array of shape (num_test,D) containing test data.

        return:
        - dists: A numpy array of shape (num_test,num_train) where dists[i,j]
            is the Euclidean distance between the ith test point and the jth training point.
        """

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test) :
            for j in range(num_train) :
                dists[i, j] = np.sqrt(np.dot(X[i] - self.X_train[j], X[i] - self.X_train[j]))
        return dists

    def compute_distances_one_loop(self, X) :

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test) :
            dists[i, :] = np.sqrt(np.sum(np.square(X[i] - self.X_train), axis=1))
        return dists

    def compute_distances_no_loops(self, X) :
        """
        Compute the distance between each test point in X and training point in self.X_train
        using no explicit loops.

        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        dists = np.sqrt(self.getNormMatrix(X, num_train).T + self.getNormMatrix(self.X_train, num_test) - 2 * np.dot(X,self.X_train.T))

        return dists

    def getNormMatrix(self, x, lines_num) :
        """
        Get a lines_num x size(x,1) matrix
        """
        return np.ones((lines_num, 1)) * np.sum(np.square(x), axis=1)

    def predict_labels(self, dists, k=1) :
        """
        given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test,num_train) where dists[i,j]
        gives the distance between the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape(num_test,) containing predicted labels for the
        test data, where y[i] is the predicted label for the test point X[i]
        """

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test) :
            closest_y = []
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            kids = np.argsort(dists[i])  # 找到距离最小的那个索引
            closest_y = self.y_train[kids[:k]]  # 找到距离最小的那个索引对应的训练数据的标签
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.
            # 找到贡献最大的那一个标签                                                 #
            count = 0
            label = 0
            for j in closest_y :
                tmp = 0
                for kk in closest_y :
                    tmp += (kk == j)
                if tmp > count :
                    count = tmp
                    label = j
            y_pred[i] = label
        return y_pred


plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# laod cifar10 data
cifar10_dir = 'E:/research/CS231n/cifar-10-python/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# print out the size of the training and test data
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


#随机显示一些样本图片
# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# num_classes = len(classes)
# samples_per_class = 7
# for y, cls in enumerate(classes) :
#     idxs = np.flatnonzero(y_train == y)  # return indices that are non-zero in the flattened version of a
#     idxs = np.random.choice(idxs,samples_per_class,replace=False)
#     for i ,idx in enumerate(idxs):
#         plt_idx=i* num_classes+y+1
#         plt.subplot(samples_per_class,num_classes,plt_idx)
#         plt.imshow(X_train[idx].astype('uint8'))
#         plt.axis('off')
#         if i==0:
#             plt.title(cls)
# plt.show()
# print("close the window to keep on")

#Subsample the data for more efficient code execution in this exercise
num_training=5000
mask=range(num_training)
X_train=X_train[mask]
y_train=y_train[mask]

num_test=500
mask=range(num_test)
X_test=X_test[mask]
y_test=y_test[mask]

#Reshape the image data into rows
X_train=np.reshape(X_train,(X_train.shape[0],-1))
X_test=np.reshape(X_test,(X_test.shape[0],-1))
print(X_train.shape,X_test.shape)

classifier=KNearestNeighbor()
classifier.train(X_train,y_train)

print("training data using two_loops...")
dists=classifier.compute_distances_two_loops(X_test)
print("training have done")
print(dists.shape)

#visualize the distance matrix:each row is a single test example and
#its distances to training examples
#plt.imshow(dists,interpolation='none')
#plt.show()

#Now implement the function predict_labels and run the code below:
# k=1
y_test_pred=classifier.predict_labels(dists,k=1)

#Compute and print the fraction of correctly predicted examples
num_correct=np.sum(y_test_pred==y_test)
accuracy=float(num_correct/num_test)
print('Got %d / %d correct => accuracy: %f' % (num_correct,num_test,accuracy))

y_test_pred=classifier.predict_labels(dists,k=5)
num_correct=np.sum(y_test_pred==y_test)
accuracy=float(num_correct/num_test)
print('Got %d / %d correct => accuracy: %f' % (num_correct,num_test,accuracy))

#Now lets speed up distance matrix computation by using partial vectorlization
#with one loop.
print('training data using one_loop')
dists_one=classifier.compute_distances_one_loop(X_test)
print('training have done')

# To ensure that our vectorized implementation is correct, we make sure that it
# agrees with the naive implementation. There are many ways to decide whether
# two matrices are similar; one of the simplest is the Frobenius norm. In case
# you haven't seen it before, the Frobenius norm of two matrices is the square
# root of the squared sum of differences of all elements; in other words, reshape
# the matrices into vectors and compute the Euclidean distance between them.
#上面简单来说就是用Frobenius范数来 判别向量化的实现与原始的two_loops生成的矩阵是否相同
difference = np.linalg.norm(dists-dists_one,ord='fro')
print('Difference was: %f ' % (difference,))
if difference <0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')

#Now implement one loop
print('training data using no_loop')
dists_no=classifier.compute_distances_no_loops(X_test)
print('training have done')

#check that the distance matrix agrees with the one we computed before:
difference=np.linalg.norm(dists-dists_no,ord='fro')
print('Difference was: %f' % (difference,))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else :
    print('Uh-oh! The distance matrices are different')

#Let's compare how fast the implementation are
def time_function(f,*args):
    """
    Call a function f with args and return the time (in seconds) that it took to execute
    """
    import time
    tic=time.time()
    f(*args)
    toc=time.time()
    return toc-tic

two_loop_time=time_function(classifier.compute_distances_two_loops,X_test)
print('Two loop version took %f seconds' % two_loop_time)

one_loop_time=time_function(classifier.compute_distances_one_loop,X_test)
print('One loop version took %f seconds ' % one_loop_time)

no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print('No loop version took %f seconds' % no_loop_time)

# you should see significantly faster performance with the fully vectorized implementation
#Cross-validation
num_folds=5
k_choices=[1,3,5,8,10,12,15,20,50,100]

X_train_folds=[]
y_train_folds=[]
# Split up the training data into folds.
X_train_folds=np.array_split(X_train,num_folds)
y_train_folds=np.array_split(y_train,num_folds)

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.

k_to_accuracies={}

# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.

for k in k_choices:
    for f in range(num_folds):
        X_train_tmp=np.array(X_train_folds[:f]+X_train_folds[f+1:])
        y_train_tmp=np.array(y_train_folds[:f]+y_train_folds[f+1:])
        X_train_tmp=X_train_tmp.reshape(-1,X_train_tmp.shape[2])
        y_train_tmp=y_train_tmp.reshape(-1)

        X_va=np.array(X_train_folds[f])
        y_va=np.array(y_train_folds[f])

        classifier.train(X_train_tmp,y_train_tmp)
        dists=classifier.compute_distances_no_loops(X_va)

        y_test_pred=classifier.predict_labels(dists,k)

        #compute and print the fraction of correclty predicted examples
        num_correct=np.sum(y_test_pred==y_va)
        accuracy=float(num_correct/y_va.shape[0])
        if( k in k_to_accuracies.keys()):
            k_to_accuracies[k].append(accuracy)
        else:
            k_to_accuracies[k]=[]
            k_to_accuracies[k].append(accuracy)

#print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k =%d, accuracy = %f' % (k,accuracy))

#plot the raw observations

for k in k_choices:
    accuracies=k_to_accuracies[k]
    plt.scatter([k]*len(accuracies),accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()




