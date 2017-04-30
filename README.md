# CNNGestureRecognizer
Gesture recognition via CNN neural network implemented in Keras + Theano + OpenCV


Key Requirements:
Python 2.7.13
OpenCV 2.4.8
Keras 2.0.2
Theano 0.9.0

Suggestion: Better to download Anaconda as it will take care of most of the other packages and easier to setup a virtual workspace to work with multiple versions of key packages like python, opencv etc.

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

