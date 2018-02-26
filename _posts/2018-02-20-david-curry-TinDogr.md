---
layout: post
title: "Image Detetcion with Transfer Learning: Under the hood of TinDogr"
date: 2018-02-20
---

# Image Detetcion with Transfer Learning and Convolutional Neural Networks: Under the hood of TinDogr

To achieve automatic matching of Tinders users based on the presence of certain dog breeds in their profiles, we have to enter the field of image classification.  Ever since the seminole paper on image classification using convolutional neural networks (CNNs) was released in 2012 ([paper link](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)), this class of algorithims have been the clear leader in the field.  Dubbed "AlexNet", Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton created a “large, deep convolutional neural network” that achieved a winning 15.4% error at the 2012 ILSVRC (ImageNet Large-Scale Visual Recognition Challenge).  For context, the closest competitor was at a measly 26.3% error!  TinDogr's patented "Dogorithm" utilizes the proven power of CNNs to detect and classify 120 dog breeds and to use this classification scheme to connect dog owners together on the Tinder dating app.  This blog post will first explore the CNN architecture and how transfer learning extends the monumental work of past image detection teams.  The second part of this post will cover my implementation usin Keras and Tensorflow in Python.

## The Frozen Layer: VGG-16
![Alt text](images/vgg16.png?raw=true "Title")
Developed in 2014, the VGG-16 CNN utlizes successive 3x3 filters and max pooling layers with two fully connected layers which handle the classification for features extracted from the convolutions step ([paper link](https://arxiv.org/pdf/1409.1556v6.pdf)).  Each filter acts an "edge detetcor of sorts:  identyfing outlines and areas of color/shade transitions.  The max pooling layers greatly reduce the dimensionality of the input, and thus the parameters needed to be learned.  The result of the convolution and max pooling layers can be thought of as a new set of features, often refered to as the "bottleneck" features. For our task of image classification and transfer learning these bottle neck features are very important.  To understand why, let's look at the final fully connected layers and remind ourselves how classification tasks are performed.

In isolation, the two fully connected(FC) layers and the final softmax output layer are nothing more than a shallow, multilayer perceptron(MLP) built for multiclass classification.  What we enter into the MLP is nothing but the features of the image classes we are trying to learn.  Tradionally this is visualized by the 2-D toy model below, which demonstrates the MLP's goal of learning the decision boundary necessary for classification.
![Alt text](images/vgg16.png?raw=true "Title")













