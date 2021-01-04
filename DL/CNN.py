# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# 32=nbr of feature detector
# input_shape resize images
# relu to remove negative pixels (black)
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
# bech ne5o  l maxpooling mouch l image donc manest7a9ouch input_shape
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
# output_dim = 128 nbr of hidden layer
classifier.add(Dense(128, activation='relu'))
# sigmoid dog+cat softmax for multiple
# output layer
classifier.add(Dense(1, activation='sigmoid'))

# Compiling the CNN
# binary_crossentropy because dog and cat
# for multiple class we use categorical_crossentropy'
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=len(training_set)//32,
                         epochs=5,
                         validation_data=test_set,
                         validation_steps=len(test_set)//32)
import numpy as np
from keras.preprocessing import image

test_image2 = image.load_img('dataset/training_set/dogs/dog.1.jpg', target_size=(64, 64))
test_image2 = image.img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis=0)
result = classifier.predict(test_image2)
training_set.class_indices

if result[0][0] == 1:
    pred4 = 'dog'

else:
    pred4 = 'cat'

print(pred4)
# =======>>  drop out mouhemma baaaaaaaaaaaaaaaaaaaaaarchaaaaaa