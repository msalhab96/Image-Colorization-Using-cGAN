# Image-Colorization-Using-cGAN
This repository is an implementation of Conditional GAN -cGAN- that converts images from greyscale to RGB.

The generator consists of an encoder and decoder, where the encoder projects a greyscale image into a vector -latent vector- and the 
decoder converts the latent vector into an RGB image; the discriminator is an encoder only that 
takes an RGB image and predict whether the input mage real or fake.

The architecture below shows the generator architecture:

![generator architecture](https://github.com/msalhab96/Image-Colorization-Using-cGAN/blob/master/images/gen_arch.jpg)

Where each block of the generator consists of:
* Convolution layer
* batch normalization 
* Relu
And the last activation function is sigmoid sigmoid 

For the discriminator, it's a stack of convolutional layers followed by a fully connected layer and sigmoid activation function.
