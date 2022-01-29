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

# Dataset

I used the Anime Faces dataset -for computational power purposes-, you can find the dataset on Kaggle [here](https://www.kaggle.com/soumikrakshit/anime-faces), the first 1000 used as a test set and the rest for training.

# Results

The models trained for ~30 epochs and there is a space for improvement, you can download the models from [here](https://drive.google.com/file/d/1f9EmKNrki3IkMMJvfnb_AclEa6-8kWSX/view?usp=sharing)

The image below shows some sample from the test data

![results](https://github.com/msalhab96/Image-Colorization-Using-cGAN/blob/master/images/results.jpg)
