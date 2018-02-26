---
layout: post
title: "Image Detetcion with Transfer Learning: Under the hood of TinDogr"
date: 2018-02-20
---

# Image Detetcion with Transfer Learning and Convolutional Neural Networks: Under the hood of TinDogr

To achieve automatic matching of Tinders users based on the presence of certain dog breeds in their profiles, we have to enter the field of image classification.  Ever since the seminole paper on image classification using convolutional neural networks (CNNs) was released in 2012 ([paper link](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)), this class of algorithims have been the clear leader in the field.  Dubbed "AlexNet", Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton created a “large, deep convolutional neural network” that achieved a winning 15.4% error at the 2012 ILSVRC (ImageNet Large-Scale Visual Recognition Challenge).  For context, the closest competitor was at a measly 26.3% error!  TinDogr's patented "Dogorithm" utilizes the proven power of CNNs to detect and classify 120 dog breeds and to use this classification scheme to connect dog owners together on the Tinder dating app.  This post will first explore the CNN architecture and how transfer learning extends the monumental work of past image detection teams.  The second part of this bog post will cover my

