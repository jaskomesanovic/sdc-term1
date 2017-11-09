# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'traffic-signs-data/train.p'
validation_file='traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print(valid)
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_valid))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import random
import numpy as np
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
# %matplotlib inline

#Unique images examples
# for index in range(0,n_classes):
#     i, = np.where(y_valid==index)
    
#     #image = X_valid[list(y_valid).index(index)].squeeze()
#     image = X_valid[i[0]].squeeze()
#     imageText = "Image =" + str(index) + ", Count = " +str(len(i))
#     plt.figure(figsize=(1,1))
#     plt.suptitle(imageText, fontsize=10)
#     plt.imshow(image, cmap="gray")
    

#from sklearn.utils import shuffle
import cv2

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def histogram_equalization(img):

    # if len(img.shape) == 2:  # grayscale image -> make a "color" image out of it
    #     img = np.dstack((img, img, img))
    # print(channels)
    # (b, g, r)=cv2.split(img)
    # img=cv2.merge([r,g,b])
    # img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #return  cv2.calcHist([img],[1],None,[256],[0,256])
    # return cv2.normalize(img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    color = [ (255,0,0),(0,255,0),(0,0,255) ]
    for ch,col in enumerate(color):
        hist_item = cv2.calcHist([img],[ch],None,[256],[0,256])  # Calculates the histogram
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX) # Normalize the value to fall below 255, to fit in image 'h'
    return img  



#X_train, y_train = shuffle(X_train, y_train)

for i in range(0,len(X_train)):
    X_train[i] = histogram_equalization(X_train[i])

