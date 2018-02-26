---
layout: post
title: "Image Detetcion with Transfer Learning: Under the hood of TinDogr"
date: 2018-02-20
---

# Image Detection with Transfer Learning and Convolutional Neural Networks: Under the hood of TinDogr

To achieve automatic matching of Tinders users based on the presence of certain dog breeds in their profiles, we have to enter the field of image classification.  Ever since the seminole paper on image classification using convolutional neural networks (CNNs) was released in 2012 ([paper link](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)), this class of algorithims have been the clear leader in the field.  Dubbed "AlexNet", Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton created a “large, deep convolutional neural network” that achieved a winning 15.4% error at the 2012 ILSVRC (ImageNet Large-Scale Visual Recognition Challenge).  For context, the closest competitor was at a measly 26.3% error!  TinDogr's patented "Dogorithm" utilizes the proven power of CNNs to detect and classify 120 dog breeds and to use this classification scheme to connect dog owners together on the Tinder dating app.  

This blog post will first explore the CNN architecture and how transfer learning extends the monumental work of past image detection teams.  The second part of this post will cover my implementation using Keras and Tensorflow in Python.

## The Frozen Layer: VGG-16
![Alt text](images/vgg16.png?raw=true "Title")

Developed in 2014, the VGG-16 CNN utlizes successive 3x3 filters and max pooling layers with two fully connected layers which handle the classification for features extracted from the convolutions step ([paper link](https://arxiv.org/pdf/1409.1556v6.pdf)).  Each filter acts an "edge detetcor of sorts:  identyfing outlines and areas of color/shade transitions.  The max pooling layers greatly reduce the dimensionality of the input, and thus the parameters needed to be learned.  The result of the convolution and max pooling layers can be thought of as a new set of features, often refered to as the "bottleneck" features. For our task of image classification and transfer learning these bottle neck features are very important.  To understand why, let's look at the final fully connected layers and remind ourselves how classification tasks are performed.

In isolation, the two fully connected(FC) layers and the final softmax output layer are nothing more than a shallow, multilayer perceptron(MLP) built for multiclass classification.  What we enter into the MLP is nothing but the features of the image classes we are trying to learn.  Tradionally this is visualized by the two parameter and class toy model below, which demonstrates the MLP's goal of learning the decision boundary necessary for classification.

![Alt text](images/classification.png?raw=true "Title")

The red and green classes above, which are separated in parameter space, are our bottleneck features.  Given a set of images classes, such as dog breeds, the VGG-16 will learn the distingushing features from the different dog breeds and encode this information into the bottleneck features.  What the network has done is taken an image and by examined it at different scales for the features that best describe it.  The network then takes these bottleneck features and simply passes it to an MLP tuned to classify a set of input features.

Now turning back to the original goal of this post-building a model that classifies dog breeds and integrates it into the Tinder app-I might be tempted to take the following approach:  build my own VGG-16 network (the architecture is widely available) and then use it to build a dog breed classifier.  However, this path is fraught with challenges.  The original VGG-16 team (Karen Simonyan and Andrew Zisserman of the University of Oxford) took two to three weeks!  And this is with access to GPUs!  I am limited by time (the TinDogr project needs to be wrapped up in its' entirety in two weeks) and by computation (a macbook pro with boring, old CPUs will be jandling all model computations).  How do I get around these constraints if I am to build a reasonably effective image classification model with minimal time and computation resources?  The answer is transfer learning.

## Transfer Learning:  Standing on the Shoulders of Giants
![Alt text](images/TF1.png?raw=true "Title")

If we take a look back at the VGG-16 architecture and the discussion on bottleneck features, something stands out.  As long as we can create bottleneck features for our different image(dog) classes then we really only need to train and tune the last two fully connected layers.  Is there an easy way to get bottleneck features for something like a dog?  Turns out there is and the answer seems fairly obvious in hindsight:  just use a previously trained CNN, like the VGG-16 model, to create a set of bottleneck features.  Someone spent weeks and lots of money on training and optimizing a model for image classification.  Of course, these legends of image detection didn't build their model for dog breed classification, but they did build it for generic image detection of hundreds of the most common objects, including dogs.  When we use a pre-trained model to create bottleneck features and then pass those features to our own fully connected layers and softmax output, we call that transfer learning.

What is important here is that none of the weights in the convolution layers are adjusted.  Back propagation is only applied to the last FC layers and thus training time is greatly decreased.  To recap, our stated goal is image classification of over 100 dog breeds.  Convolutional neural networks are currently the model best suited to this task, but training time of the most effective models can often exceed a week, even with the full power of multiple GPUs.  A transfer learning technique with the successful VGG-16 pre-trained weights is employed in order create a robust dog breed image classifier.  I know turn to a discussion of the code and pipeline used for TinDogr.

## Importing the data
The Stanford machine learning repository is the source of our training data: 120 dog breeds spread over ~20 thousand photos.  It's important to note that some breeds contain much more data than others(the max class has ~350 photos, while the min class has ~50 photos).  Neural networks are notoriously "hungry" in that they need lots of data to perform well.  Current wisdom places the number at ~2K images per class in order to achieve acceptable classification accuracy.  We will address this data deficiency by generating more training data with data augmentation techniques, but before that we must split the data into train and validation sets.

### Creating the Training and Validation Datasets
Data is currently has the structure:
```
Images   
└───dog breed folder1
│   │   file011.png
│   │   file012.png
│   │   ...
└───dog breed folder2
    │   file021.txt
    │   file022.txt
    │   ...
```
Below is a short function that creates train and test directories with a specified % split of photos for each breed.

```python
def train_test_dir_split(root_dir, percent_train):
    '''
    Takes in a single root dir of image classes and splits into test and train root dirs.
    Specify the percent split that each class/sub-dir should have(one % for whole set)
    '''

    train_dir = root_dir+'/train/'
    test_dir  = root_dir+'/test/'
    my_dirs = [i for i in os.listdir(root_dir)]
     
    # make if not present
    if not os.path.isdir(root_dir+'/train/'):
    	os.system('mkdir '+train_dir)
        os.system('mkdir '+test_dir) 
    
    for i,my_dir in enumerate(my_dirs):
        file_list = [name for name in os.listdir(root_dir+my_dir)]
        num_files = len(file_list)
       
        if not os.path.isdir(root_dir+'/train/'+my_dir):
            os.system('mkdir '+train_dir+my_dir)
            os.system('mkdir '+test_dir+my_dir)

        # get fraction
        test_frac = 1-percent_train
    	test_num = int(len(file_list)*test_frac)
    
        # cp test/train files to test dir
	   	for i,file in enumerate(file_list
            if i < test_num:
                os.system('cp '+root_dir+my_dir+'/'+file+' '+test_dir+my_dir)
            else:
                os.system('cp '+root_dir+my_dir+'/'+file+' '+train_dir+my_dir)        
```

## Data Augmentation and Keras Generators
Now that I have train and validation datasets created I can use data augmentation to increase the number of images per class.  Luckily Keras has the ImageDataGenerator class, which handles the import, augment, and general flow of images through thr VGG-16 network.  Below is the code for importing the pre-trained VGG-16 weights and creating the bottleneck features (data augmentation is part of the pipeline thanks to Keras).  The full script can be seen [here](https://github.com/dcurry09/TinDogr/blob/master/python/train_CNN_script.py).

```python
def save_bottleneck_features():
        ''' Saves bottleneck features for test/train sets for use later on.
            Separating this step speeds up later optimization.
        '''
	model = applications.VGG16(weights = 'imagenet', include_top = False)

	datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range = 40,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            shear_range = 0.2,
            zoom_range = 0.2,
	 )

	train_generator = datagen.flow_from_directory(
           train_data_dir,
           target_size = (IMG_SIZE, IMG_SIZE),
           batch_size = batch_size,
   	   shuffle = False
	)
	
	test_generator = datagen.flow_from_directory(
           test_data_dir,
           target_size = (IMG_SIZE, IMG_SIZE),
           batch_size = batch_size,
           class_mode = None,
           shuffle = False
	)
	
	# Do not retrain if already present
	if not os.path.isfile('weights/bottleneck_features_train.npy'):
            bottleneck_features_train = model.predict_generator(train_generator, train_samples//batch_size, verbose=1)
	    bottleneck_features_test = model.predict_generator(test_generator, test_samples//batch_size, verbose=1)		    np.save(open('weights/bottleneck_features_train.npy','wb'), bottleneck_features_train)
	    np.save(open('weights/bottleneck_features_test.npy','wb'), bottleneck_features_test)
        else:
            bottleneck_features_train = np.load('weights/bottleneck_features_train.npy')
            bottleneck_features_test = np.load('weights/bottleneck_features_test.npy')

        # I will also need the target classes and labels for post-bottleneck training
	train_y = to_categorical(train_generator.classes)
        test_y = to_categorical(test_generator.classes)
        train_labels = train_generator.class_indices
        np.save(open('weights/train_y.npy','wb'), train_y)
        np.save(open('weights/train_labels.npy','wb'), train_labels)
        np.save(open('weights/test_y.npy','wb'), test_y)
    
	return bottleneck_features_train, bottleneck_features_test, train_y, train_labels, test_y

```
The bottleneck features for our dog breed classes have now been created and saved for future use.  Besides having these features for visualization and exploration, I also have reduced the time it will take to optimize the next step: the transfer of the bottleneck features to the fully connected classification layers.  If we had not saved the bottleneck features ahead of time then I have had to pass each image through the VGG-16 conv layers during each stage of optimization.

## Fully Connected Layers
Now I will create shallow MLP and plug the output of the VGG-16 (the bottleneck features) into the input of my new layers.


```python
def train_top_model(train_data, train_Y, test_data, test_Y):
   ''' Training of FC layers with bottleneck features as inputs
   '''
   model = Sequential()
   model.add(Flatten(input_shape = train_data.shape[1:]))
   model.add(Dense(2056, activation='relu'))
   model.add(Dropout(0.2))
   model.add(Dense(1028, activation='relu'))
   model.add(Dense(NUM_CLASSES, activation='softmax'))

   opt = optimizers.SGD(lr=0.01)
   model.compile(optimizer = opt,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

   checkpointer = ModelCheckpoint(filepath='model.best.hdf5', verbose=1, save_best_only=False)

   model.fit(train_data, train_Y,
             epochs=epochs,
             batch_size=batch_size,
             validation_data = [test_data, test_Y],
             callbacks = [checkpointer])

   model.save_weights(top_model_weights_path)

```

Keras has made the transfer learning technique straightforward and flexible.  What I have shown so far is only half the battle however.  We may have built a dog breed classifier, but it still need to be integrated into the Tinder dating app ecosystem.

## Putting Everything Online
The API I will be using to connect to Tinder is called Pynder and can be accessed on Github [here](https://github.com/charliewolf/pynder).  Flask and AWS will be used to create and host TinDogr repsectively.  The main Flask script can be seen on Github [here](https://github.com/dcurry09/TinDogr/blob/master/python/flaskr.py).  The work flow for detection of a compatible dog breed and automatic swiping is:

1) Connect to Tinder using your Facebook credentials
2) Scan area for all Tinder users with matching criteria(age, gender, distance)
3) For each user inspect each profile image for a dog
4) Classify any dogs present, assign a size, and compare to your dogs size
5) Swipe right(left) if dog breed is(not) compatible size
