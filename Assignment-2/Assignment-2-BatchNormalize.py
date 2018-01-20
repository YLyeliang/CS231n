import time
import numpy as np
import matplotlib.pyplot as plt

from fc_net import *
from data_utils import get_CIFAR10_data
from gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from layers import affine_backward, affine_forward, relu_forward, relu_backward
from layer_utils import affine_relu_backward, affine_relu_forward
from layers import svm_loss, softmax_loss
from solver import Solver



data = get_CIFAR10_data()

hidden_dims = [100, 100, 100, 100, 100]
num_train = 5000
small_data = {
    'X_train': data['X_train'][:num_train],
    'y_train': data['y_train'][:num_train],
    'X_val': data['X_val'],
    'y_val': data['y_val'],
}
bn_solvers = {}
solvers = {}
weight_scales = np.logspace(-4, 0, num = 20)
for i, weight_scale in enumerate(weight_scales):
    print('Running weight scale %d / %d' % (i + 1, len(weight_scales)))
    bn_model = FullyConnectedNet(hidden_dims, weight_scale = weight_scale, use_batchnorm = True)
    model = FullyConnectedNet(hidden_dims, weight_scale = weight_scale, use_batchnorm = False)

    bn_solver = Solver(bn_model, small_data,
                       num_epochs = 10, batch_size = 100,
                       update_rule = 'adam',
                       optim_config = {'learning_rate': 1e-3, },
                       verbose = False, print_every = 1000)
    bn_solver.train()
    bn_solvers[weight_scale] = bn_solver

    solver = Solver(model, small_data,
                    num_epochs = 10, batch_size = 100,
                    update_rule = 'adam',
                    optim_config = {'learning_rate': 1e-3, },
                    verbose = False, print_every = 1000)
    solver.train()
    solvers[weight_scale] = solver

# Plot results of weight scale experiment
best_train_accs, bn_best_train_accs = [], []
best_val_accs, bn_best_val_accs = [], []
final_train_loss, bn_final_train_loss = [], []

for ws in weight_scales:
    best_train_accs.append(max(solvers[ws].train_acc_history))
    bn_best_train_accs.append(max(bn_solvers[ws].train_acc_history))

    best_val_accs.append(max(solvers[ws].val_acc_history))
    bn_best_val_accs.append(max(bn_solvers[ws].val_acc_history))

    final_train_loss.append(np.mean(solvers[ws].loss_history[-100:]))
    bn_final_train_loss.append(np.mean(bn_solvers[ws].loss_history[-100:]))

plt.subplot(3, 1, 1)
plt.title('Best val accuracy vs weight initialization scale')
plt.xlabel('Weight initialization scale')
plt.ylabel('Best val accuracy')
plt.semilogx(weight_scales, best_val_accs, '-o', label = 'baseline')
plt.semilogx(weight_scales, bn_best_val_accs, '-o', label = 'batchnorm')
plt.legend(ncol = 2, loc = 'lower right')

plt.subplot(3, 1, 2)
plt.title('Best train accuracy vs weight initialization scale')
plt.xlabel('Weight initialization scale')
plt.ylabel('Best training accuracy')
plt.semilogx(weight_scales, best_train_accs, '-o', label = 'baseline')
plt.semilogx(weight_scales, bn_best_train_accs, '-o', label = 'batchnorm')
plt.legend(loc = 'upper left')

plt.subplot(3, 1, 3)
plt.title('Final training loss vs weight initialization scale')
plt.xlabel('Weight initialization scale')
plt.ylabel('Final training loss')
plt.semilogx(weight_scales, final_train_loss, '-o', label = 'baseline')
plt.semilogx(weight_scales, bn_final_train_loss, '-o', label = 'batchnorm')
plt.legend(loc = 'upper left')

plt.gcf().set_size_inches(10, 15)
plt.show()