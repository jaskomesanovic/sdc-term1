## Writeup Template
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/ChessboarCorners.jpg "Image"
[image2]: ./output_images/calibration_distortion_original.jpg "Image2"
[image3]: ./output_images/calibration_distortion_Fixed.jpg "Image3"
[image4]: ./output_images/calibration_distortion_Diff.jpg "Image4"
[image5]: ./output_images/test_image_distortion_original.jpg "Image5"
[image6]: ./output_images/test_image_distortion_Fixed.jpg "Image6"
[image7]: ./output_images/test_image_distortion_Diff.jpg "Image7"
[image8]: ./output_images/before_line_search.jpg "Image8"
[image9]: ./output_images/after_line_search.jpg "Image9"
[image10]: ./output_images/final_img_mask.jpg "Image10"
[image11]: ./output_images/persp_transform_original.jpg "Image11"
[image12]: ./output_images/persp_transform_warped.jpg "Image12"
[image13]: ./output_images/straight_lines1.jpg.jpg "Image13"
[image14]: ./output_images/straight_lines2.jpg.jpg "Image14"
[image15]: ./output_images/test1.jpg.jpg "Image15"
[image16]: ./output_images/test2.jpg.jpg "Image16"
[image17]: ./output_images/test3.jpg.jpg "Image17"
[image18]: ./output_images/test4.jpg.jpg "Image18"
[image19]: ./output_images/test5.jpg.jpg "Image19"
[image20]: ./output_images/test6.jpg.jpg "Image20"
[image21]: ./output_images/distortion1.png "Image21"
[image22]: ./output_images/masked_warped.png "Image21"
[image23]: ./output_images/histogram.png "Image21"



[video1]: ./output_images/project_video.mp4 "Video"
[video2]: ./output_images/challenge_video.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation. 

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the third code cell of the IPython notebook located in "./LaneLines.ipynb". 

I start by preparing "object points", which will be the (x, y,z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.

An example of the checssboar corners is
![alt text][image1]

 `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

Original image
![alt text][image2]

Undistorted image
![alt text][image3]

Difference
![alt text][image4]
### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Distortion correction is explained in the previous step and it can be found in the 3 and 4 cell of the IPython.
Example of the image is shown below together with the difference among original and undistorted image. We can notice that biggest differences are at the edges of the pictures:

![alt text][image21]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (IPython cells #5 and #6). Since in the picture we have a lot of noise and we are interested only in the lane lines which are usually white or yellow color we need to find appropriate way to identify only point of interest and remove the noise. In the cell #5 we can see methods for the color filtering.

To make it more robust, we also compute a mask based on gradients. In particular, we use the Sobel operator seen in the lectures, using the OpenCv function cv2.Sobel. We have experimented with gradients in X and Y directions independently, gradient magnitude and direction. The implementation appears in cell #6. Combined usage of the filters is in the cell #8 The conclusions are:

* Sobel in X direction is extremely useful since the lane lines are vertical. Sobel Y can detect most of them as well, but returns extra undesirable gradients for example when having shadows across the road.

* Gradient magnitude combines sobel X and Y, therefore keeping the problems of Sobel Y.

* Gradient direction is extremely noisy and doesn't allow us to better extract the lane lines.


Here's an example of my output for this step.

![alt text][image10]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `get_birds_eye_view()`, which appears cell # 9.  The `get_birds_eye_view()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
y_horizon = 470
src_pts_ = ((x1, img.shape[0]),
           (x2, img.shape[0]),
           (717, y_horizon),
           (566, y_horizon))

off = 100 # Horizontal offset to have more space and better estimate sharp curves
dst_pts_ = ((x1 + off, img.shape[0]),
           (x2 - off, img.shape[0]),
           (x2 - off, 0),
           (x1 + off, 0)) 
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 200, 720      | 320, 720      |
| 1120, 720     | 1020, 720     |
| 566, 470      | 300, 0        |
| 717, 470      | 1020, 0       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image11]
![alt text][image12]

We used the OpenCV functions `cv2.getPerspectiveTransform` and `cv2.warpPerspective` for this purpose.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

When we receive the first video frame, we have no information about the lane lines in the image. Therefore we must perform a search without prior assumptions. We will work with the birds-eye view image (img_warped), passed through the combined_mask() function, as explained before.

The implemented approach is as follows:

* Discover the starting point of the line, in the bottom of the image.
* Follow the line all the way up to the top of the image, using a sliding window technique.

In the cell #10 we can see how the starting point x is chosen. We are using histogram of the lower part of the picture to identify where the lane starts. Example is like bellow

![alt text][image22]
![alt text][image23]

The next step is to place a box around this starting point, extract the non-zero pixels inside it, and then move it upwards following the line, all the way up to the top of the image.

The sliding window is moved as follows:

It moves the amount size_y in the vertical direction, where size_y is the size of the window.
If the window contained pixels, it moves towards the mean x position of those pixels. Otherwise it moves the same amount as in the previous step, assuming that the line has the same curvature in the image (cell #11).

![alt text][image9]

Once we have the pixels for each line, we can perform line fitting, where we simply fit a second-order polynomial to the stored x and y data points. This is performed using the function np.polyfit(y, x, 2)(cell #11 and `find_lines()` class `Lane` cell #12 ).

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

It is in the class `Lane` method `find_lines`

```python
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])

right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

curvature = (left_curverad + right_curverad)/2
```

as we can see we need to have y value of the image (y_eval) and polynomials in the world space. Formula for the curvature is shown on the lecture. ym_per_pix represents meters per pixels. We have assumed that our point of interested is 700 px and that represent about 15 meters so we can calculate our ratio.
After we calculate curvatures for both lines we take average of both to be final curvature number.

For the position of the car in the image we used code from below

```python
image_center = binary_warped.shape[1]/2
lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
offset = (lane_center - image_center) * xm_per_pix
```
![alt text][image13]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell #12 in the function `execute()` class `LanePipeline`.


```python
# Create an image to draw the lines on
warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

cv2.putText(undist,"Road curvature:%.1f m" % lane.line_left.radius_of_curvature,(100,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255, 255, 255),2)
cv2.putText(undist,"Distance:%.1f m" % lane.line_left.line_base_pos,(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255, 255, 255),2)

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
```

Here is an example of my result on a test image:

![alt text][image18]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/20tQmnkhKyI)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have to say that this project for me was the most exciting. I spent a lot of time on the color masks trying to tweak  the thresholds. While working on the validation part of the found lanes I saw very well difference after using the meaned results. Lane selection was much more precise and didn't showed huge oscillations but downside was that sometimes because of mean value it didn't perfectly fit to the line. It was noticed in the upper part of the lane.

I didn't like that we had to define our points for the perspective transform. IMHO that is the weak point of the algorithm. Also all videos are recorded on the nice weather and there is not so much noise, but on the rainy or snowing days this algorithm would be not reliable. I guess this is just one step for the resilient solution for the lane findings and that in practice some more sensors must be used in order to validate position of the car.






