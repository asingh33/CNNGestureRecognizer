# CNNGestureRecognizer
Gesture recognition via CNN neural network implemented in Keras + Theano + OpenCV


Key Requirements:
Python 2.7.13
OpenCV 2.4.8
Keras 2.0.2
Theano 0.9.0

Suggestion: Better to download Anaconda as it will take care of most of the other packages and easier to setup a virtual workspace to work with multiple versions of key packages like python, opencv etc.


# Repo contents
- **trackgesture.py** : The main script launcher. This file contains all the code for UI options and OpenCV code to capture camera contents. This script internally calls interfaces to gestureCNN.py.
- **gestureCNN.py** : This script file holds all the CNN specific code to create CNN model, load the weight file (if model is pretrained), train the model using image samples present in **./imgfolder_b**, visualize the feature maps at different layers of NN (of pretrained model) for a given input image present in **./imgs** folder.
- **imgfolder_b** : This folder contains all the 4015 gesture images I took in order to train the model. Only reason I had to provide instead of weight file is GitHub's restriction for >100 MB file upload. **So in case you are trying to download/pull my repo then you must first train the model either using this imgfolder_b contents or your own.**
- **_imgs_** - This is an optional folder of few sample images that one can use to visualize the feature maps at different layers. These are few sample images from imgfolder_b only.
- **_ori_4015imgs_acc.png_** : This is just a pic of a plot depicting model accuracy Vs validation data accuracy after I trained it.
- **_ori_4015imgs_loss.png_** : This is just a pic of a plot depicting model loss Vs validation loss after I training.


# Usage
This application comes with CNN model to recognize upto 5 pretrained gestures:
- OK
- PEACE
- STOP
- PUNCH
- NOTHING (ie when none of above gestures are input)

This application provides following functionalities:
- Prediction : Which allows the app to guess the user's gesture against pretrained gestures. App can dump the prediction data to the console terminal or to a json file directly which can be used to plot real time prediction bar chart (you can use my other script - https://github.com/asingh33/LivePlot)
- New Training : Which allows the user to retrain the NN model. User can change the model architecture or add/remove new gestures. This app has inbuilt options to allow the user to create new image samples of user defined gestures if required.
- Visualization : Which allows the user to see feature maps of different NN layers for a given input gesture image. Interesting to see how NN works and learns things.


# Demo 
Youtube link - https://www.youtube.com/watch?v=CMs5cn65YK8

![](https://j.gifs.com/X6zwYm.gif)


# ToDo
Add more details of the implementation. Add some references as well.


# Conclusion
So where to go from here? Well I thought of testing out the responsiveness of NN predictions and games are good benchmark. On MAC I dont have any games installed but then this Chrome Browser Dino Jump game came handy. So I bound the 'Punch' gesture with jump action of the Dino character. Basically can work with any other gesture but felt Punch gesture is easy. Stop gesture was another candidate.
Well here is how it turned out :)
YouTube link - https://www.youtube.com/watch?v=lnFPvtCSsLA&t=49s

![](https://j.gifs.com/58pxVx.gif)
