# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
import cv2
import math


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  

    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    if len(img.shape) == 2:  # grayscale image -> make a "color" image out of it
        img = np.dstack((img, img, img))

 
    for line in lines:
        if len(line[0]) == 4:
            for x1, y1, x2, y2 in line:
                if x1 > 0 and x1 < img.shape[1] and \
                    y1 > 0 and y1 < img.shape[0] and \
                    x2 > 0 and x2 < img.shape[1] and \
                    y2 > 0 and y2 < img.shape[0]:
                    # print(x1, y1, x2, y2)
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    # print(img)
    return img


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
        []), minLineLength=min_line_len, maxLineGap=max_line_gap)
   
    line_img = extract_lane_lines(img.shape, lines)

    return line_img

# Python 3 has support for cool math symbols.


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def extract_lane_lines(imshape, lines):

    slope_min = 0.5
    slope_max = 1.0

    # Left line
    m1 = np.array([])
    b1 = np.array([])

    # Right line
    m2 = np.array([])
    b2 = np.array([])

    yMin = imshape[0]
    yMax = imshape[0] - 1

    for line in lines:
        # print(line)
        for x_left_1, y_left_1, x_right_1, y_right_1 in line:
            m = (y_right_1 - y_left_1) / (x_right_1 - x_left_1)
            b = y_left_1 - m * x_left_1

            if abs(m) > slope_min and abs(m) < slope_max:
                if m > 0:
                    m1 = np.append(m1, m)
                    b1 = np.append(b1, b)
                else:
                    m2 = np.append(m2, m)
                    b2 = np.append(b2, b)

                yMin = min([yMin, y_left_1, y_right_1])

    m1 = np.mean(m1)
    b1 = np.mean(b1)

    m2 = np.mean(m2)
    b2 = np.mean(b2)

    y_left_1 = yMax
    x_left_1 = round((y_left_1 - b1) / m1)

    y_left_2 = yMin
    x_left_2 = round((y_left_2 - b1) / m1)

    y_right_1 = yMax
    x_right_1 = round((y_right_1 - b2) / m2)

   
    y_right_2 = yMin
    x_right_2 = round((y_right_2 - b2) / m2)

    linePoints = np.array(
        [[[x_left_1, y_left_1, x_left_2, y_left_2]], [[x_right_1, y_right_1, x_right_2, y_right_2]]]).astype(int)
    return linePoints


def return_y_position(y_position):
    # It needs to be lover half of the picture
    return y_position * 0.6



def get_houg_lines(img, masked_edges):
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # 15 minimum number of votes (intersections in Hough grid cell)
    threshold = 30
    min_line_length = 80  # minimum number of pixels making up a line
    max_line_gap = 150    # maximum gap in pixels between connectable line segments
    

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = hough_lines(masked_edges, rho, theta, threshold,
                        min_line_length, max_line_gap)
    return lines

def define_vertices(imshape):
    vertices = np.array([[(0, imshape[0]), (450, return_y_position(
        imshape[0])), (500, 330), (imshape[1], imshape[0])]], dtype=np.int32)
    return vertices

def apply_canny(blur_gray):
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    return edges

def apply_gaussian_smoothing(gray_image):
    kernel_size = 5
    blur_gray = gaussian_blur(gray_image, kernel_size)
    return blur_gray


def pipeline(img):

    imshape = img.shape
    gray_image = grayscale(img)

    blur_gray = apply_gaussian_smoothing(gray_image)

    edges = apply_canny(blur_gray)

    vertices = define_vertices(imshape)

    masked_edges = region_of_interest(edges, vertices)

    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    lines = get_houg_lines(img, masked_edges)

    lane_image = draw_lines(line_image, lines)
    	
    return weighted_img(lane_image,img)


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result=pipeline(image)
    return result

def test_images():
    test_images_directory = "test_images"
    result_dir= "test_images_output"

    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    for filename in  os.listdir(test_images_directory):
        image_path= os.path.join(test_images_directory, filename)
        img = mpimg.imread(image_path)

        result_image=process_image(img)

        mpimg.imsave(os.path.join(result_dir, filename), result_image)


def test_videos(video_name):
    test_video_directory="test_videos"
    result_video_output=os.path.join("test_videos_output", video_name)
   
    if not os.path.isdir("test_videos_output"):
            os.mkdir("test_videos_output")
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip(os.path.join(test_video_directory, video_name))
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(result_video_output, audio=False)


test_images()

videos=["solidYellowLeft.mp4","solidWhiteRight.mp4"]

for video in videos:
    test_videos(video)
