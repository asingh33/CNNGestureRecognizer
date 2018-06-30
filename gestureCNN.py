#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 01:01:43 2017

@author: abhisheksingh
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

# We require this for Theano lib ONLY. Remove it for TensorFlow usage
from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
#import matplotlib.pyplot as plt
import os
import theano
from PIL import Image
# SKLEARN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json

import cv2
import matplotlib
#matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


# input image dimensions
img_rows, img_cols = 200, 200

# number of channels
# For grayscale use 1 value and for color images use 3 (R,G,B channels)
img_channels = 1


# Batch_size to train
batch_size = 32

## Number of output classes (change it accordingly)
## eg: In my case I wanted to predict 4 types of gestures (Ok, Peace, Punch, Stop)
## NOTE: If you change this then dont forget to change Labels accordingly
nb_classes = 5

# Number of epochs to train (change it accordingly)
nb_epoch = 15  #25

# Total number of convolutional filters to use
nb_filters = 32
# Max pooling
nb_pool = 2
# Size of convolution kernel
nb_conv = 3

#%%
#  data
path = "./"
path1 = "./gestures"    #path of folder of images

## Path2 is the folder which is fed in to training model
path2 = './imgfolder_b'

WeightFileName = ["ori_4015imgs_weights.hdf5","bw_4015imgs_weights.hdf5","bw_2510imgs_weights.hdf5","./bw_weight.hdf5","./final_c_weights.hdf5","./semiVgg_1_weights.hdf5","/new_wt_dropout20.hdf5","./weights-CNN-gesture_skinmask.hdf5"]

# outputs
output = ["OK", "NOTHING","PEACE", "PUNCH", "STOP"]
#output = ["PEACE", "STOP", "THUMBSDOWN", "THUMBSUP"]



#%%
# This function can be used for converting colored img to Grayscale img
# while copying images from path1 to path2
def convertToGrayImg(path1, path2):
    listing = os.listdir(path1)
    for file in listing:
        if file.startswith('.'):
            continue
        img = Image.open(path1 +'/' + file)
        #img = img.resize((img_rows,img_cols))
        grayimg = img.convert('L')
        grayimg.save(path2 + '/' +  file, "PNG")

#%%
def modlistdir(path):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        #This check is to ignore any hidden files/folders
        if name.startswith('.'):
            continue
        retlist.append(name)
    return retlist


# Load CNN model
def loadCNN(wf_index):
    global get_output
    model = Sequential()
    
    
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                        padding='valid',
                        input_shape=(img_channels, img_rows, img_cols)))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    '''
    
    model.add(ZeroPadding2D((1,1),input_shape=(img_channels, img_rows, img_cols)))
    model.add(Conv2D(nb_filters , (nb_conv, nb_conv), activation='relu'))
    #model.add(ZeroPadding2D((1,1)))
    #model.add(Conv2D(nb_filters , (nb_conv, nb_conv), activation='relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.2))
    
    #model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(nb_filters , (nb_conv, nb_conv), activation='relu'))
    #model.add(ZeroPadding2D((1,1)))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    ##
    #model.add(Conv2D(nb_filters , (nb_conv, nb_conv), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=(2,2)))
    
    model.add(Dropout(0.3))
    model.add(Flatten())
    ###
    #model.add(Dense(128))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    '''
    
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    
    # Model summary
    model.summary()
    # Model conig details
    model.get_config()
    
    from keras.utils import plot_model
    plot_model(model, to_file='new_model.png', show_shapes = True)
    

    if wf_index >= 0:
        #Load pretrained weights
        fname = WeightFileName[int(wf_index)]
        print("loading ", fname)
        model.load_weights(fname)
    
    layer = model.layers[11]
    get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
    
    
    return model

# This function does the guessing work based on input images
def guessGesture(model, img):
    global output, get_output
    #Load image and flatten it
    image = np.array(img).flatten()
    
    # reshape it
    image = image.reshape(img_channels, img_rows,img_cols)
    
    # float32
    image = image.astype('float32') 
    
    # normalize it
    image = image / 255
    
    # reshape for NN
    rimage = image.reshape(1, img_channels, img_rows, img_cols)
    
    # Now feed it to the NN, to fetch the predictions
    #index = model.predict_classes(rimage)
    #prob_array = model.predict_proba(rimage)
    
    prob_array = get_output([rimage, 0])[0]
    
    #print prob_array
    
    d = {}
    i = 0
    for items in output:
        d[items] = prob_array[0][i] * 100
        i += 1
    
    # Get the output with maximum probability
    import operator
    
    guess = max(d.items(), key=operator.itemgetter(1))[0]
    prob  = d[guess]

    if prob > 70.0:
        #print guess + "  Probability: ", prob

        #Enable this to save the predictions in a json file,
        #Which can be read by plotter app to plot bar graph
        #dump to the JSON contents to the file
        
        with open('gesturejson.txt', 'w') as outfile:
            json.dump(d, outfile)

        return output.index(guess)

    else:
        return 1

#%%
def initializers():
    imlist = modlistdir(path2)
    
    image1 = np.array(Image.open(path2 +'/' + imlist[0])) # open one image to get size
    #plt.imshow(im1)
    
    m,n = image1.shape[0:2] # get the size of the images
    total_images = len(imlist) # get the 'total' number of images
    
    # create matrix to store all flattened images
    immatrix = np.array([np.array(Image.open(path2+ '/' + images).convert('L')).flatten()
                         for images in sorted(imlist)], dtype = 'f')
    

    
    print(immatrix.shape)
    
    input("Press any key")
    
    #########################################################
    ## Label the set of images per respective gesture type.
    ##
    label=np.ones((total_images,),dtype = int)
    
    samples_per_class = total_images / nb_classes
    print("samples_per_class - ",samples_per_class)
    s = 0
    r = samples_per_class
    for classIndex in range(nb_classes):
        label[s:r] = classIndex
        s = r
        r = s + samples_per_class
    
    '''
    # eg: For 301 img samples/gesture for 4 gesture types
    label[0:301]=0
    label[301:602]=1
    label[602:903]=2
    label[903:]=3
    '''
    
    data,Label = shuffle(immatrix,label, random_state=2)
    train_data = [data,Label]
     
    (X, y) = (train_data[0],train_data[1])
     
     
    # Split X and y into training and testing sets
     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
     
    X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)
     
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
     
    # normalize
    X_train /= 255
    X_test /= 255
     
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, X_test, Y_train, Y_test



def trainModel(model):

    # Split X and y into training and testing sets
    X_train, X_test, Y_train, Y_test = initializers()

    # Now start the training of the loaded model
    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                 verbose=1, validation_split=0.2)

    visualizeHis(hist)

    ans = input("Do you want to save the trained weights - y/n ?")
    if ans == 'y':
        filename = input("Enter file name - ")
        fname = path + str(filename) + ".hdf5"
        model.save_weights(fname,overwrite=True)
    else:
        model.save_weights("newWeight.hdf5",overwrite=True)

    # Save model as well
    # model.save("newModel.hdf5")
#%%

def visualizeHis(hist):
    # visualizing losses and accuracy

    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(nb_epoch)

    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    #plt.style.use(['classic'])

    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)

    plt.show()

#%%
def visualizeLayers(model, img, layerIndex):
    imlist = modlistdir('./imgs')
    if img <= len(imlist):
        
        image = np.array(Image.open('./imgs/' + imlist[img - 1]).convert('L')).flatten()
        
        ## Predict
        guessGesture(model,image)
        
        # reshape it
        image = image.reshape(img_channels, img_rows,img_cols)
        
        # float32
        image = image.astype('float32')
        
        # normalize it
        image = image / 255
        
        # reshape for NN
        input_image = image.reshape(1, img_channels, img_rows, img_cols)
    else:
        X_train, X_test, Y_train, Y_test = initializers()
        
        # the input image
        input_image = X_test[:img+1]
    
    
    
        
    # visualizing intermediate layers
    #output_layer = model.layers[layerIndex].output
    #output_fn = theano.function([model.layers[0].input], output_layer)
    #output_image = output_fn(input_image)
    
    if layerIndex >= 1:
        visualizeLayer(model,img,input_image, layerIndex)
    else:
        tlayers = len(model.layers[:])
        print("Total layers - {}".format(tlayers))
        for i in range(1,tlayers):
             visualizeLayer(model,img, input_image,i)

#%%
def visualizeLayer(model, img, input_image, layerIndex):

    layer = model.layers[layerIndex]
    
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
    activations = get_activations([input_image, 0])[0]
    output_image = activations
    
    
    ## If 4 dimensional then take the last dimension value as it would be no of filters
    if output_image.ndim == 4:
        # Rearrange dimension so we can plot the result
        o1 = np.rollaxis(output_image, 3, 1)
        output_image = np.rollaxis(o1, 3, 1)
        
        print("Dumping filter data of layer{} - {}".format(layerIndex,layer.__class__.__name__))
        filters = len(output_image[0,0,0,:])
        
        fig=plt.figure(figsize=(8,8))
        # This loop will plot the 32 filter data for the input image
        for i in range(filters):
            ax = fig.add_subplot(6, 6, i+1)
            #ax.imshow(output_image[img,:,:,i],interpolation='none' ) #to see the first filter
            ax.imshow(output_image[0,:,:,i],'gray')
            #ax.set_title("Feature map of layer#{} \ncalled '{}' \nof type {} ".format(layerIndex,
            #                layer.name,layer.__class__.__name__))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
        plt.tight_layout()
        #plt.show()
        fig.savefig("img_" + str(img) + "_layer" + str(layerIndex)+"_"+layer.__class__.__name__+".png")
        #plt.close(fig)
    else:
        print("Can't dump data of this layer{}- {}".format(layerIndex, layer.__class__.__name__))


