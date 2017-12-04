import os
import csv
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

DROPOUT_PROB = 0.2
BATCH_SIZE= 64

def read_sample_from_file():
    samples = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples

def generator(samples, batch_size=64, is_validation=False):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        #(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './data/'+batch_sample[i].split('\\')[-1].strip()
                    center_image = cv2.imread(name)
                    images.append(center_image)

                center_angle = float(batch_sample[3])
                correction= 0.2
                angles.append(center_angle)
                angles.append(center_angle + correction)
                angles.append(center_angle - correction)

            #For validation set we don't need to augument the data
            if(is_validation):
                X_train = np.array(images)
                y_train = np.array(angles)
            else:
                X_train, y_train = augument_data(images, angles)

            yield sklearn.utils.shuffle(X_train, y_train)

def augument_data(images, angles):
    augumented_images=[]
    augumented_angles =[]
    for image, angle in zip(images, angles):
        augumented_images.append(image)
        augumented_angles.append(angle)
        flipped_image=cv2.flip(image,1)
        flipped_angle = float(angle) * -1.0
        augumented_images.append(flipped_image)
        augumented_angles.append(flipped_angle)            
    X_train = np.array(augumented_images)
    y_train = np.array(augumented_angles)
    return X_train, y_train



def training_model(dropout_prob):
    weight_init='glorot_uniform'
    padding = 'valid'
    ch, row, col = 3, 160, 320  # Trimmed image format

    model= Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5,
                        input_shape=(row, col, ch ),
                        output_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, border_mode=padding,init = weight_init, subsample = (2, 2), activation='relu'))
    model.add(Convolution2D(36,5,5,subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48,5,5,subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten()) 
    model.add(Dropout(dropout_prob))
    model.add(Dense(100))
    model.add(Dropout(dropout_prob))
    model.add(Dense(50))
    model.add(Dropout(dropout_prob))
    model.add(Dense(10))
    model.add(Dropout(dropout_prob))
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mse', optimizer='adam')
   
    return model


samples = read_sample_from_file()
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE, is_validation=True)


model = training_model(DROPOUT_PROB)
model.fit_generator(train_generator, 
                samples_per_epoch= len(train_samples)*3.84, 
                validation_data=validation_generator, 
                nb_val_samples=len(validation_samples), 
                nb_epoch=3)

model.save('model.h5')

exit()

