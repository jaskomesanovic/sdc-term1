---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/1p.png "Traffic Sign 1"
[image5]: ./examples/2p.png "Traffic Sign 2"
[image6]: ./examples/3p.png "Traffic Sign 3"
[image7]: ./examples/4p.png "Traffic Sign 4"
[image8]: ./examples/5p.png "Traffic Sign 5"
[image9]: ./examples/TestImages.PNG "Test images"
[image10]: ./examples/AugumentedTrainingDistribution.png "Augumented Training Data Distribution"
[image11]: ./examples/TrainingDistribution.png "Training Data Distribution"
[image12]: ./examples/TestDistribution.png "Test Data Distribution"
[image13]: ./examples/Lossfunction.png "Training Loss Function"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jaskomesanovic/sdc-term1/tree/master/Term%201/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how  one example of the different signs looks 

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Original training data set is having distribution like on the picture below.

![alt text][image11]

It is obvious that shown data set is not having normal distribution and that according to it model will behave the same. Which means that it will have more information about certain signs and identify them better. 

My idea was to do data augmentation and to make a little bit better data distribution. After doing data augmentation distribution was like on the picture

![alt text][image10]

Test Data distribution is similar to the original training data set.

![alt text][image12]

In original data set number of pictures per sign was from 200 to 2000. In the extended data set it is from 5800 to 7000. Data augmentation has improved our data model significantly which will be explained after.

During the data augmentation several picture transformation was used like
* Rotation
* Translation
* Shear
* Brightness adjustment

All this techniques enriched our model which will simulate better different conditions and situation on the road. All the code can be seen in the 4th cell of the Ipython notebook

After merging original and augmented data set I did additional data preparation. 
As a first step, I decided to convert the images to grayscale because  that will improve training performance. In this case our image matrix will have shape 32,32,1 and we will not lose image information.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it helps during the learning process. CNN are learning their weights by continually adding gradient error vectors (multiplied by a learning rate) computed from backpropagation to various weight matrices throughout the network as training examples are passed through.  
If we didn't normalize  our input training vectors, the ranges of our distributions of feature values would likely be different for each feature, and thus the learning rate would cause corrections in each dimension that would differ (proportionally speaking) from one another. We might be overcompensating a correction in one weight dimension while undercompensating in another.

This is non-ideal as we might find ourselves in a oscillating state or in a slow moving (traveling too slow to get to a better maxima) state.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is following LeNet architecture and it is consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale normalized image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding  outputs 14x14x6			|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding  outputs 5x5x16	
| Flatten  	      	    | input = 5x5x16, outputs = 400
| Fully connected		| input = 400, output = 120                     |
| RELU					|												|
| Fully connected		| input = 120, output = 84                     |
| RELU					|												|
| Fully connected		| input = 84, output = 43                     |

The code can be found in the 6th cell of the notebook.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The model was trained using the 

* Cross entropy with logits
* Adam optimizer as suggested in the course because it is robust enough for this type scenarios and performs better than GradientDescentOptimizer
* Batch size is 128 for training and testing. With this number it can fit to the GPU memory (4GB).
* Learning rate is 0.003. I started with suggested learning rate of 0.001 but while tuning parameters I had good results with this result.
* Number of epochs is 30. I found that my model within this number have good performance and do not show signs of overfitting. When increasing number of epochs execution time was getting bigger of course and training accuracy was even decreasing.

   

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

* What was the first architecture that was tried and why was it chosen?

First architecture was LeNet because it provides a good baseline for image classification problems. 
        
* What were some problems with the initial architecture?

Applying LeNet to the training data set gave me accuracy of 89 % which was not good enough
I modified model to accept input with a depth of 3 instead of 1 because images are grayscale.
I changed the length of the output to the 43 instead of 10
In my opinion  problem was also small number of epochs which result in underfitting. Another thing was also quality of the input data. I was spending a lot of time with testing how the input data are affecting final result. At the end I increased number of pictures per sign and also augment the data.
        
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
 
I have added dropout to the all  layers but result was not good. The best results I got with the dropout in the Convolutional layers and not in the fully connected layers.
 
* Which parameters were tuned? How were they adjusted and why?

Number of epochs: 10 epochs were not enough. When I increased the data set it was necessary to increase number of epochs. I stopped at 30 because my model was showing signs of overfitting for higher numbers.
Learning rate: I was putting smaller and bigger learning rate then original 0.001 but at the end I had accuracy 96% with 0.003
Dropout: During the training I was trying to have 0.5. This was extremely powerful to make the network more robust and prevent from overfitting
 
My final model results were:
* training set accuracy of 0.966
* validation set accuracy of 0.944 
* test set accuracy of 0.930

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image9] 

Images that I choose are having very good quality and I expected to have high percentage of accuracy.
They are clean and sign is in the central position without a lot of noise. Shape of the signs is clearly visible and seen from frontal position without any rotation.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 60 km/h	      		| 60 km/h					 				    |
| Yield					| Yield											|
| Stop Sign      		| Stop sign   									|
| 30 km/h	      		| 30 km/h					 				    |
| No entry     			| No entry 										|



The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. Like I previously explained test pictures are having very good quality and this result is expected.Test data set is also too small (5 images) for defining final accuracy.  

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 21th cell of the Ipython notebook.

Image predictions percentage can be visible on the next pictures.

![alt text][image4]![alt text][image5]
![alt text][image6]![alt text][image7]
![alt text][image8]