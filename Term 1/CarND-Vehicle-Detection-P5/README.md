**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[image1]: ./output_images/vehicle_non_vehicle.jpg
[image2]: ./output_images/image_non_vehicle-hog.jpg
[image11]: ./output_images/image_vehicle-hog.jpg
[image3]: ./output_images/feature_normalization_histogram.jpg
[image4]: ./output_images/search_window.jpg

[image5]: ./output_images/test1_heatmap.png
[image6]: ./output_images/test2_heatmap.png
[image7]: ./output_images/test3_heatmap.png
[image8]: ./output_images/test4_heatmap.png
[image9]: ./output_images/test5_heatmap.png
[image10]: ./output_images/test6_heatmap.png

[image12]: ./output_images/image_boxes.png

[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation. 


### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point. 

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 5th code cell of the IPython notebook. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=8` and `cells_per_block=2`:


![alt text][image2]
![alt text][image11]

#### 2. Explain how you settled on your final choice of HOG parameters.

The motivation for choosing previous  parameters was mostly trial and error to get the best trade-off between computational complexity and accuracy. I first performed quick tests on the sandbox in the lecture 32. Search and Classify. I tried values within the range that was suggested in the documentation for the hog function.

I found that pixels_per_cell = 16 did a pretty good job at removing false positives. However some true positives were removed too, so I decided to switch back to 8 pixels per cells. Another reason for choosing 8 over 16 is that for a 64x64 image we would only get 4 blocks, which makes it very likely that vehicle is not detected if it's not well centered in the image.

A little bit of experimentation with cells_per_block  and number_of_bins  was also performed, but I didn't find significant differences in performance. Therefore I used the default values shown in the lectures and documentation, which seemed to perform just fine.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM as explained in the next few cells using the following approach:

I took params  that yielded the best overall prediction accuracy against the test set.

I then invoke #extract_features which extracts HOG features from each training image for both the cars and not cars datasets passing the same parameter to each.

I then stack both cars and not cars datasets into a single array called X which correspond to my training features.

Using sklearn.preprocessing.StandardScaler(), I normalize my feature vectors for training my classifier. Here is example of the normalization

![alt text][image3]

Then I apply the same scaling to each of the feature vectors.

Next, I created my training labels by using np.hstack which assigns the label 1 for each item from the cars training set and the label 0 for each item in the notcars training set.

Then I split up the data into randomized 80% training and 20% test sets using sklearn.model_selection.train_test_split. This automatically shuffles my dataset.

Using sklearn.svm.LinearSVC, I fit my training features and labels to the model.

Finally, I run a prediction against my model and print some statistics to the console below the 11th Jupyter Notebook cell.


I wanted to test which color space would be the best and which hog channels would give me the best results so I trained data like  a test  with method `slide_test`


```python
color_space_opt=["RGB", "HSV", "LUV", "HLS", "YUV", "YCrCb"]
hog_channels =[0, 1, 2, "ALL"]

for color_space in color_space_opt:
   for hog_channel in hog_channels:
       slide_test(color_space,hog_channel)
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used method `slide_window` provided on the lecture from Udacity with
with (64,64)px size window with scale from 1.0 to 2.0 in 5 steps  with overlap of 0.75. 
I first started with scale 1.5 but pipeline performed bad and it was not being able to  identify the same car on different position in the video since the size of the car dependes of the distance from our view. After some tests I decide to choose parameters that I mentioned at beginning.

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image12]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/VjHYeU3OQE0)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 




Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]




---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are still some false positives and wobbling in the project video. I  succeeded to  stabilize boxes with doing average of heat maps over the frames but still that was not enough. Because of my time constraints I was not able to test it more and maybe to tweak it better.

What I didn't like and have a feeling that it can be better done  is that we are spending too much time on computation. Because of the search windows we are scanning image several time which at the end result that pipeline is slow and I don't think it can be used in live.

Since we are depending on our training data for successful classification I know that algorithm will suffer if we have rain, snow or too much light.



