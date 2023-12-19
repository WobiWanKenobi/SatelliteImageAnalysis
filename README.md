# Satellite Image Classification

## General

This python script creates a [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) (cnn) to classify optical satellite images from the EUROSat dataset, that was created during ESA's Sentinal-2 mission of it's earth observing Copernicus Program, in 10 different classes. These classes could possibly be extended after training, by training the network on the EUROSat mulitband dataset, but for now this script is using the rgb dataset, to only take on the classification task. Possible output labels are forest, highway, river and industrial to name a few. Pytorch Lightning was used

## Input and Model Architecture

The network takes the images and scales them down to a resolution of 16 by 16 pixels, this still preserves enough information for the classification while also providing a decent training speed. The neural networks core consists out of 3 convolutional layers, for optimizing the loss function the [Adam](https://arxiv.org/abs/1412.6980) alglorithm is used, comined with a learn rate scheduler and weight decay.

## Performance

Right now the network achieves a classification accuracy during it's validation step of just over 90%. This is mainly achieved due to optimizing the training process by data augmentation: The modification of the existing dataset, e.g rescaling, solarizing or adding color jitter to the original data, etc. This helps the network to generate better results, because now the original image library of 27.000 labeled and georeferenced images gets expanded a few times. To visualize performance Tensorboard is used

## What's next?

To further improve this scripts additional steps need to be made, like adding an automatic trainer to find the optimal hyperparameters, for achieving even better validation results. Also the multiband dataset could be implemented in the training, to extract all possible information and make classification better and more diverse. 

