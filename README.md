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
