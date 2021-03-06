
# coding: utf-8

# # Generative Adversarial Networks (GANs)
#
# So far in CS231N, all the applications of neural networks that we have explored have been **discriminative models** that take an input and are trained to produce a labeled output. This has ranged from straightforward classification of image categories to sentence generation (which was still phrased as a classification problem, our labels were in vocabulary space and we’d learned a recurrence to capture multi-word labels). In this notebook, we will expand our repetoire, and build **generative models** using neural networks. Specifically, we will learn how to build models which generate novel images that resemble a set of training images.
#
# ### What is a GAN?
#
# In 2014, [Goodfellow et al.](https://arxiv.org/abs/1406.2661) presented a method for training generative models called Generative Adversarial Networks (GANs for short). In a GAN, we build two different neural networks. Our first network is a traditional classification network, called the **discriminator**. We will train the discriminator to take images, and classify them as being real (belonging to the training set) or fake (not present in the training set). Our other network, called the **generator**, will take random noise as input and transform it using a neural network to produce images. The goal of the generator is to fool the discriminator into thinking the images it produced are real.
#
# We can think of this back and forth process of the generator ($G$) trying to fool the discriminator ($D$), and the discriminator trying to correctly classify real vs. fake as a minimax game:
# $$\underset{G}{\text{minimize}}\; \underset{D}{\text{maximize}}\; \mathbb{E}_{x \sim p_\text{data}}\left[\log D(x)\right] + \mathbb{E}_{z \sim p(z)}\left[\log \left(1-D(G(z))\right)\right]$$
# where $x \sim p_\text{data}$ are samples from the input data, $z \sim p(z)$ are the random noise samples, $G(z)$ are the generated images using the neural network generator $G$, and $D$ is the output of the discriminator, specifying the probability of an input being real. In [Goodfellow et al.](https://arxiv.org/abs/1406.2661), they analyze this minimax game and show how it relates to minimizing the Jensen-Shannon divergence between the training data distribution and the generated samples from $G$.
#
# To optimize this minimax game, we will aternate between taking gradient *descent* steps on the objective for $G$, and gradient *ascent* steps on the objective for $D$:
# 1. update the **generator** ($G$) to minimize the probability of the __discriminator making the correct choice__.
# 2. update the **discriminator** ($D$) to maximize the probability of the __discriminator making the correct choice__.
#
# While these updates are useful for analysis, they do not perform well in practice. Instead, we will use a different objective when we update the generator: maximize the probability of the **discriminator making the incorrect choice**. This small change helps to allevaiate problems with the generator gradient vanishing when the discriminator is confident. This is the standard update used in most GAN papers, and was used in the original paper from [Goodfellow et al.](https://arxiv.org/abs/1406.2661).
#
# In this assignment, we will alternate the following updates:
# 1. Update the generator ($G$) to maximize the probability of the discriminator making the incorrect choice on generated data:
# $$\underset{G}{\text{maximize}}\;  \mathbb{E}_{z \sim p(z)}\left[\log D(G(z))\right]$$
# 2. Update the discriminator ($D$), to maximize the probability of the discriminator making the correct choice on real and generated data:
# $$\underset{D}{\text{maximize}}\; \mathbb{E}_{x \sim p_\text{data}}\left[\log D(x)\right] + \mathbb{E}_{z \sim p(z)}\left[\log \left(1-D(G(z))\right)\right]$$
#
# ### What else is there?
# Since 2014, GANs have exploded into a huge research area, with massive [workshops](https://sites.google.com/site/nips2016adversarial/), and [hundreds of new papers](https://github.com/hindupuravinash/the-gan-zoo). Compared to other approaches for generative models, they often produce the highest quality samples but are some of the most difficult and finicky models to train (see [this github repo](https://github.com/soumith/ganhacks) that contains a set of 17 hacks that are useful for getting models working). Improving the stabiilty and robustness of GAN training is an open research question, with new papers coming out every day! For a more recent tutorial on GANs, see [here](https://arxiv.org/abs/1701.00160). There is also some even more recent exciting work that changes the objective function to Wasserstein distance and yields much more stable results across model architectures: [WGAN](https://arxiv.org/abs/1701.07875), [WGAN-GP](https://arxiv.org/abs/1704.00028).
#
#
# GANs are not the only way to train a generative model! For other approaches to generative modeling check out the [deep generative model chapter](http://www.deeplearningbook.org/contents/generative_models.html) of the Deep Learning [book](http://www.deeplearningbook.org). Another popular way of training neural networks as generative models is Variational Autoencoders (co-discovered [here](https://arxiv.org/abs/1312.6114) and [here](https://arxiv.org/abs/1401.4082)). Variational autoencoders combine neural networks with variational inference to train deep generative models. These models tend to be far more stable and easier to train but currently don't produce samples that are as pretty as GANs.
#
# Example pictures of what you should expect (yours might look slightly different):
#
# ![caption](gan_outputs_tf.png)
#

# ## Setup

# In[ ]:


import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# A bunch of utility functions

def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))

    return

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def count_params():
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(x.get_shape().as_list()) for x in tf.global_variables()])
    return param_count


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

answers = np.load('gan-checks-tf.npz')


# ## Dataset
#  GANs are notoriously finicky with hyperparameters, and also require many training epochs. In order to make this assignment approachable without a GPU, we will be working on the MNIST dataset, which is 60,000 training and 10,000 test images. Each picture contains a centered image of white digit on black background (0 through 9). This was one of the first datasets used to train convolutional neural networks and it is fairly easy -- a standard CNN model can easily exceed 99% accuracy.
#
# To simplify our code here, we will use the TensorFlow MNIST wrapper, which downloads and loads the MNIST dataset. See the [documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py) for more information about the interface. The default parameters will take 5,000 of the training examples and place them into a validation dataset. The data will be saved into a folder called `MNIST_data`.
#
# **Heads-up**: The TensorFlow MNIST wrapper returns images as vectors. That is, they're size (batch, 784). If you want to treat them as images, we have to resize them to (batch,28,28) or (batch,28,28,1). They are also type np.float32 and bounded [0,1].

# In[ ]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./cs231n/datasets/MNIST_data', one_hot=False)

# show a batch
show_images(mnist.train.next_batch(16)[0])
plt.show()

# ## LeakyReLU
# In the cell below, you should implement a LeakyReLU. See the [class notes](http://cs231n.github.io/neural-networks-1/) (where alpha is small number) or equation (3) in [this paper](http://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf). LeakyReLUs keep ReLU units from dying and are often used in GAN methods (as are maxout units, however those increase model size and therefore are not used in this notebook).
#
# HINT: You should be able to use `tf.maximum`



def leaky_relu(x, alpha = 0.01):
    """Compute the leaky ReLU activation function.

    Inputs:
    - x: TensorFlow Tensor with arbitrary shape
    - alpha: leak parameter for leaky ReLU

    Returns:
    TensorFlow Tensor with the same shape as x
    """
    # TODO: implement leaky ReLU

    # If x is below 0 returns alpha*x else it will return x.
    activation = tf.maximum(x, alpha * x)

    return activation


# Test your leaky ReLU implementation. You should get errors < 1e-10

# In[ ]:


def test_leaky_relu(x, y_true):
    tf.reset_default_graph()
    with get_session() as sess:
        y_tf = leaky_relu(tf.constant(x))
        y = sess.run(y_tf)
        print('Maximum error: %g' % rel_error(y_true, y))


test_leaky_relu(answers['lrelu_x'], answers['lrelu_y'])


# ## Random Noise
# Generate a TensorFlow `Tensor` containing uniform noise from -1 to 1 with shape `[batch_size, dim]`.

# In[ ]:


def sample_noise(batch_size, dim):
    """Generate random uniform noise from -1 to 1.

    Inputs:
    - batch_size: integer giving the batch size of noise to generate
    - dim: integer giving the dimension of the the noise to generate

    Returns:
    TensorFlow Tensor containing uniform noise in [-1, 1] with shape [batch_size, dim]
    """
    # TODO: sample and return noise

    random_noise=tf.random_uniform([batch_size,dim],-1,1)

    return random_noise


# Make sure noise is the correct shape and type:

# In[ ]:


def test_sample_noise():
    batch_size = 3
    dim = 4
    tf.reset_default_graph()
    with get_session() as sess:
        z = sample_noise(batch_size, dim)
        # Check z has the correct shape
        assert z.get_shape().as_list() == [batch_size, dim]
        # Make sure z is a Tensor and not a numpy array
        assert isinstance(z, tf.Tensor)
        # Check that we get different noise for different evaluations
        z1 = sess.run(z)
        z2 = sess.run(z)
        assert not np.array_equal(z1, z2)
        # Check that we get the correct range
        assert np.all(z1 >= -1.0) and np.all(z1 <= 1.0)
        print("All tests passed!")


test_sample_noise()


# ## Discriminator
# Our first step is to build a discriminator. You should use the layers in `tf.layers` to build the model.
# All fully connected layers should include bias terms.
#
# Architecture:
#  * Fully connected layer from size 784 to 256
#  * LeakyReLU with alpha 0.01
#  * Fully connected layer from 256 to 256
#  * LeakyReLU with alpha 0.01
#  * Fully connected layer from 256 to 1
#
# The output of the discriminator should have shape `[batch_size, 1]`, and contain real numbers corresponding to the scores that each of the `batch_size` inputs is a real image.

# In[ ]:


def discriminator(x):
    """Compute discriminator score for a batch of input images.

    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]

    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score
    for an image being real for each input image.
    """
    with tf.variable_scope("discriminator"):
        # TODO: implement architecture

        fc1=tf.layers.dense(inputs = x,units = 256,activation = leaky_relu)
        fc2=tf.layers.dense(inputs = fc1,units = 256,activation = leaky_relu)
        logits=tf.layers.dense(inputs = fc2,units = 1)

        return logits


# Test to make sure the number of parameters in the discriminator is correct:

# In[ ]:


def test_discriminator(true_count = 267009):
    tf.reset_default_graph()
    with get_session() as sess:
        y = discriminator(tf.ones((2, 784)))
        cur_count = count_params()
        if cur_count != true_count:
            print('Incorrect number of parameters in discriminator. {0} instead of {1}. Check your achitecture.'.format(
                cur_count, true_count))
        else:
            print('Correct number of parameters in discriminator.')


test_discriminator()



# ## Generator
# Now to build a generator. You should use the layers in `tf.layers` to construct the model. All fully connected layers should include bias terms.
#
# Architecture:
#  * Fully connected layer from tf.shape(z)[1] (the number of noise dimensions) to 1024
#  * ReLU
#  * Fully connected layer from 1024 to 1024
#  * ReLU
#  * Fully connected layer from 1024 to 784
#  * TanH (To restrict the output to be [-1,1])

# In[ ]:


def generator(z):
    """Generate images from a random noise vector.

    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]

    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    with tf.variable_scope("generator"):
        # TODO: implement architecture

        fc1=tf.layers.dense(inputs = z,units = 1024,activation =tf.nn.relu)
        fc2=tf.layers.dense(inputs = fc1,units = 1024,activation = tf.nn.relu)
        img=tf.layers.dense(inputs = fc2,units = 784,activation = tf.nn.tanh)

        return img


# Test to make sure the number of parameters in the generator is correct:

# In[ ]:


def test_generator(true_count = 1858320):
    tf.reset_default_graph()
    with get_session() as sess:
        y = generator(tf.ones((1, 4)))
        cur_count = count_params()
        if cur_count != true_count:
            print('Incorrect number of parameters in generator. {0} instead of {1}. Check your achitecture.'.format(
                cur_count, true_count))
        else:
            print('Correct number of parameters in generator.')


test_generator()



# # GAN Loss
#
# Compute the generator and discriminator loss. The generator loss is:
# $$\ell_G  =  -\mathbb{E}_{z \sim p(z)}\left[\log D(G(z))\right]$$
# and the discriminator loss is:
# $$ \ell_D = -\mathbb{E}_{x \sim p_\text{data}}\left[\log D(x)\right] - \mathbb{E}_{z \sim p(z)}\left[\log \left(1-D(G(z))\right)\right]$$
# Note that these are negated from the equations presented earlier as we will be *minimizing* these losses.
#
# **HINTS**: Use [tf.ones_like](https://www.tensorflow.org/api_docs/python/tf/ones_like) and [tf.zeros_like](https://www.tensorflow.org/api_docs/python/tf/zeros_like) to generate labels for your discriminator. Use [sigmoid_cross_entropy loss](https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits) to help compute your loss function. Instead of computing the expectation, we will be averaging over elements of the minibatch, so make sure to combine the loss by averaging instead of summing.

# In[ ]:


def gan_loss(logits_real, logits_fake):
    """Compute the GAN loss.

    Inputs:
    - logits_real: Tensor, shape [batch_size, 1], output of discriminator
        Log probability that the image is real for each real image
    - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
        Log probability that the image is real for each fake image

    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """
    # TODO: compute D_loss and G_loss

    # Target label vector for generator loss and used in discriminator loss.
    true_lables=tf.ones_like(logits_fake)

    # Discriminator loss has 2 parts:how well it classifies real images and how well it classifies fake images.
    real_image_loss=tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real,labels = true_lables)
    fake_image_loss=tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,labels = 1-true_lables)

    # Combine and average losses over the batch
    D_loss=real_image_loss+fake_image_loss
    D_loss=tf.reduce_mean(D_loss)

    # Generator is trying to make the discriminator output 1 for all its images.
    # So we use our target label vector of onese for computing generator loss.
    G_loss=tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,labels = true_lables)

    # Average generator loss over the batch.
    G_loss=tf.reduce_mean(G_loss)

    return D_loss, G_loss


# Test your GAN loss. Make sure both the generator and discriminator loss are correct. You should see errors less than 1e-5.

# In[ ]:


def test_gan_loss(logits_real, logits_fake, d_loss_true, g_loss_true):
    tf.reset_default_graph()
    with get_session() as sess:
        d_loss, g_loss = sess.run(gan_loss(tf.constant(logits_real), tf.constant(logits_fake)))
    print("Maximum error in d_loss: %g" % rel_error(d_loss_true, d_loss))
    print("Maximum error in g_loss: %g" % rel_error(g_loss_true, g_loss))


test_gan_loss(answers['logits_real'], answers['logits_fake'],
              answers['d_loss_true'], answers['g_loss_true'])


# # Optimizing our loss
# Make an `AdamOptimizer` with a 1e-3 learning rate, beta1=0.5 to mininize G_loss and D_loss separately. The trick of decreasing beta was shown to be effective in helping GANs converge in the [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) paper. In fact, with our current hyperparameters, if you set beta1 to the Tensorflow default of 0.9, there's a good chance your discriminator loss will go to zero and the generator will fail to learn entirely. In fact, this is a common failure mode in GANs; if your D(x) learns to be too fast (e.g. loss goes near zero), your G(z) is never able to learn. Often D(x) is trained with SGD with Momentum or RMSProp instead of Adam, but here we'll use Adam for both D(x) and G(z).

# In[ ]:


# TODO: create an AdamOptimizer for D_solver and G_solver
def get_solvers(learning_rate = 1e-3, beta1 = 0.5):
    """Create solvers for GAN training.

    Inputs:
    - learning_rate: learning rate to use for both solvers
    - beta1: beta1 parameter for both solvers (first moment decay)

    Returns:
    - D_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
    - G_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
    """
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1)

    return D_solver, G_solver



# ## Putting it all together
# Now just a bit of Lego Construction.. Read this section over carefully to understand how we'll be composing the generator and discriminator

# In[ ]:


tf.reset_default_graph()

# number of images for each batch
batch_size = 128
# our noise dimension
noise_dim = 96

# placeholder for images from the training dataset
x = tf.placeholder(tf.float32, [None, 784])
# random noise fed into our generator
z = sample_noise(batch_size, noise_dim)
# generated images
G_sample = generator(z)

with tf.variable_scope("") as scope:
    #scale images to be -1 to 1
    logits_real = discriminator(preprocess_img(x))
    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    logits_fake = discriminator(G_sample)

# Get the list of variables for the discriminator and generator
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

# get our solver
D_solver, G_solver = get_solvers()

# get our loss
D_loss, G_loss = gan_loss(logits_real, logits_fake)

# setup training steps
D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)
D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')


# # Training a GAN!
# Well that wasn't so hard, was it? In the iterations in the low 100s you should see black backgrounds, fuzzy shapes as you approach iteration 1000, and decent shapes, about half of which will be sharp and clearly recognizable as we pass 3000. In our case, we'll simply train D(x) and G(z) with one batch each every iteration. However, papers often experiment with different schedules of training D(x) and G(z), sometimes doing one for more steps than the other, or even training each one until the loss gets "good enough" and then switching to training the other.

# In[ ]:


# a giant helper function
def run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step, show_every = 250,
              print_every = 50, batch_size = 128, num_epoch = 10):
    """Train a GAN for a certain number of epochs.

    Inputs:
    - sess: A tf.Session that we want to use to run our data
    - G_train_step: A training step for the Generator
    - G_loss: Generator loss
    - D_train_step: A training step for the Generator
    - D_loss: Discriminator loss
    - G_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for generator
    - D_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for discriminator
    Returns:
        Nothing
    """
    # compute the number of iterations we need
    max_iter = int(mnist.train.num_examples * num_epoch / batch_size)
    for it in range(max_iter):
        # every show often, show a sample result
        if it % show_every == 0:
            samples = sess.run(G_sample)
            fig = show_images(samples[:16])
            plt.show()
            print()
        # run a batch of data through the network
        minibatch, minbatch_y = mnist.train.next_batch(batch_size)
        _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict = {x: minibatch})
        _, G_loss_curr = sess.run([G_train_step, G_loss])

        # print loss every so often.
        # We want to make sure D_loss doesn't go to 0
        if it % print_every == 0:
            print('Iter: {}, D: {:.4}, G:{:.4}'.format(it, D_loss_curr, G_loss_curr))
    print('Final images')
    samples = sess.run(G_sample)

    fig = show_images(samples[:16])
    plt.show()


# #### Train your GAN! This should take about 10 minutes on a CPU, or less than a minute on GPU.

# In[ ]:


with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step)



# # Least Squares GAN
# We'll now look at [Least Squares GAN](https://arxiv.org/abs/1611.04076), a newer, more stable alternative to the original GAN loss function. For this part, all we have to do is change the loss function and retrain the model. We'll implement equation (9) in the paper, with the generator loss:
# $$\ell_G  =  \frac{1}{2}\mathbb{E}_{z \sim p(z)}\left[\left(D(G(z))-1\right)^2\right]$$
# and the discriminator loss:
# $$ \ell_D = \frac{1}{2}\mathbb{E}_{x \sim p_\text{data}}\left[\left(D(x)-1\right)^2\right] + \frac{1}{2}\mathbb{E}_{z \sim p(z)}\left[ \left(D(G(z))\right)^2\right]$$
#
#
# **HINTS**: Instead of computing the expectation, we will be averaging over elements of the minibatch, so make sure to combine the loss by averaging instead of summing. When plugging in for $D(x)$ and $D(G(z))$ use the direct output from the discriminator (`score_real` and `score_fake`).

# In[ ]:


def lsgan_loss(score_real, score_fake):
    """Compute the Least Squares GAN loss.

    Inputs:
    - score_real: Tensor, shape [batch_size, 1], output of discriminator
        score for each real image
    - score_fake: Tensor, shape[batch_size, 1], output of discriminator
        score for each fake image

    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """
    # TODO: compute D_loss and G_loss

    true_labels=tf.ones_like(score_fake)
    fake_image_loss=tf.reduce_mean((score_real-true_labels)**2)
    real_image_loss=tf.reduce_mean(score_fake**2)
    D_loss=0.5*(fake_image_loss+real_image_loss)
    G_loss=0.5*tf.reduce_mean((score_fake-true_labels)**2)

    return D_loss, G_loss


# Test your LSGAN loss. You should see errors less than 1e-7.

# In[ ]:


def test_lsgan_loss(score_real, score_fake, d_loss_true, g_loss_true):
    with get_session() as sess:
        d_loss, g_loss = sess.run(
            lsgan_loss(tf.constant(score_real), tf.constant(score_fake)))
    print("Maximum error in d_loss: %g" % rel_error(d_loss_true, d_loss))
    print("Maximum error in g_loss: %g" % rel_error(g_loss_true, g_loss))


test_lsgan_loss(answers['logits_real'], answers['logits_fake'],
                answers['d_loss_lsgan_true'], answers['g_loss_lsgan_true'])

# Create new training steps so we instead minimize the LSGAN loss:

# In[ ]:


D_loss, G_loss = lsgan_loss(logits_real, logits_fake)
D_train_step = D_solver.minimize(D_loss, var_list = D_vars)
G_train_step = G_solver.minimize(G_loss, var_list = G_vars)

# In[ ]:


with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step)


# # INLINE QUESTION 1:
# Describe how the visual quality of the samples changes over the course of training. Do you notice anything about the distribution of the samples? How do the results change across different training runs?

# ** (Write Your Answer In This Cell) **

# # Deep Convolutional GANs
# In the first part of the notebook, we implemented an almost direct copy of the original GAN network from Ian Goodfellow. However, this network architecture allows no real spatial reasoning. It is unable to reason about things like "sharp edges" in general because it lacks any convolutional layers. Thus, in this section, we will implement some of the ideas from [DCGAN](https://arxiv.org/abs/1511.06434), where we use convolutional networks as our discriminators and generators.
#
# #### Discriminator
# We will use a discriminator inspired by the TensorFlow MNIST classification [tutorial](https://www.tensorflow.org/get_started/mnist/pros), which is able to get above 99% accuracy on the MNIST dataset fairly quickly. *Be sure to check the dimensions of x and reshape when needed*, fully connected blocks expect [N,D] Tensors while conv2d blocks expect [N,H,W,C] Tensors.
#
# Architecture:
# * 32 Filters, 5x5, Stride 1, Leaky ReLU(alpha=0.01)
# * Max Pool 2x2, Stride 2
# * 64 Filters, 5x5, Stride 1, Leaky ReLU(alpha=0.01)
# * Max Pool 2x2, Stride 2
# * Flatten
# * Fully Connected size 4 x 4 x 64, Leaky ReLU(alpha=0.01)
# * Fully Connected size 1

# In[ ]:


def discriminator(x):
    """Compute discriminator score for a batch of input images.

    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]

    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score
    for an image being real for each input image.
    """
    with tf.variable_scope("discriminator"):
        # TODO: implement architecture
        unflatten = tf.reshape(x, shape = [-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(inputs = unflatten, kernel_size = 5, strides = 1, filters = 32,
                                 activation = leaky_relu)
        maxpool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = 2, strides = 2)
        conv2 = tf.layers.conv2d(inputs = maxpool1, kernel_size = 5, strides = 1, filters = 64, activation = leaky_relu)
        maxpool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = 2, strides = 2)
        flatten = tf.reshape(maxpool2, shape = [-1, 1024])
        fc1 = tf.layers.dense(inputs = flatten, units = 1024, activation = leaky_relu)
        logits = tf.layers.dense(inputs = fc1, units = 1)
        return logits


test_discriminator(1102721)


# #### Generator
# For the generator, we will copy the architecture exactly from the [InfoGAN paper](https://arxiv.org/pdf/1606.03657.pdf). See Appendix C.1 MNIST. See the documentation for [tf.nn.conv2d_transpose](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose). We are always "training" in GAN mode.
#
# Architecture:
# * Fully connected of size 1024, ReLU
# * BatchNorm
# * Fully connected of size 7 x 7 x 128, ReLU
# * BatchNorm
# * Resize into Image Tensor
# * 64 conv2d^T (transpose) filters of 4x4, stride 2, ReLU
# * BatchNorm
# * 1 conv2d^T (transpose) filter of 4x4, stride 2, TanH

# In[ ]:


def generator(z):
    """Generate images from a random noise vector.

    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]

    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    with tf.variable_scope("generator"):
        # TODO: implement architecture
        fc1 = tf.layers.dense(inputs = z, units = 1024, activation = tf.nn.relu)
        bn1 = tf.layers.batch_normalization(inputs = fc1, training = True)
        fc2 = tf.layers.dense(inputs = bn1, units = 7 * 7 * 128, activation = tf.nn.relu)
        bn2 = tf.layers.batch_normalization(inputs = fc2, training = True)
        reshaped = tf.reshape(bn2, shape = [-1, 7, 7, 128])
        conv_transpose1 = tf.layers.conv2d_transpose(inputs = reshaped, filters = 64, kernel_size = 4, strides = 2,
                                                     activation = tf.nn.relu,
                                                     padding = 'same')
        bn3 = tf.layers.batch_normalization(inputs = conv_transpose1, training = True)
        conv_transpose2 = tf.layers.conv2d_transpose(inputs = bn3, filters = 1, kernel_size = 4, strides = 2,
                                                     activation = tf.nn.tanh,
                                                     padding = 'same')

        img = tf.reshape(conv_transpose2, shape = [-1, 784])
        return img


test_generator(6595521)

# We have to recreate our network since we've changed our functions.

# In[ ]:


tf.reset_default_graph()

batch_size = 128
# our noise dimension
noise_dim = 96

# placeholders for images from the training dataset
x = tf.placeholder(tf.float32, [None, 784])
z = sample_noise(batch_size, noise_dim)
# generated images
G_sample = generator(z)

with tf.variable_scope("") as scope:
    # scale images to be -1 to 1
    logits_real = discriminator(preprocess_img(x))
    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    logits_fake = discriminator(G_sample)

# Get the list of variables for the discriminator and generator
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

D_solver, G_solver = get_solvers()
D_loss, G_loss = gan_loss(logits_real, logits_fake)
D_train_step = D_solver.minimize(D_loss, var_list = D_vars)
G_train_step = G_solver.minimize(G_loss, var_list = G_vars)
D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')

# ### Train and evaluate a DCGAN
# This is the one part of A3 that significantly benefits from using a GPU. It takes 3 minutes on a GPU for the requested five epochs. Or about 50 minutes on a dual core laptop on CPU (feel free to use 3 epochs if you do it on CPU).

# In[ ]:


with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step, num_epoch = 5)