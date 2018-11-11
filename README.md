# DCGAN

Implementation of Deep Convolutional Generative Adversarial Networks (DCGANs) using PyTorch.

A DCGAN is a direct extension of the GAN described above, except that it explicitly uses convolutional and convolutional-transpose layers 
in the discriminator and generator, respectively. It was first described by Radford et. al. in the paper 
[Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434).
The discriminator is made up of strided convolution layers, batch norm layers, and LeakyReLU activations.
The input is a 3x64x64 input image and the output is a scalar probability that the input is from the real data distribution.
The generator is comprised of convolutional-transpose layers, batch norm layers, and ReLU activations.
The input is a latent vector, z, that is drawn from a standard normal distribution and the output is a 3x64x64 RGB image.
The strided conv-transpose layers allow the latent vector to be transformed into a volume with the same shape as an image.

# Dataset

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

The dataset can be downloaded from here : [CIFAR10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

```Note : The dataset can be downloaded by running the cells in the notebook too ```

# Requirements

1. PyTorch
2. Torchvision
3. Python

```utils.py has been taken from online resources.It serves the purpose of visualizing the performance of the model```

```I recommend to make use of GPUs or Cloud Platforms to train the model```

```The code for utilizing GPU is also included```
