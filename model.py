import csv
import cv2
import numpy as np
import os
import math

# parameters to be tuned
correction = 0.2
epochs = 3

DATA_PATH = '../data/'
images = []
measurements = []
def _read_in_data (path, steering, correction):
  image = cv2.imread(path)
  images.append(image)
  measurement = float(line[3])
  measurements.append(measurement)

def _get_path (basepath, raw_path):
  return basepath + '/IMG/' + raw_path.split('/')[-1]

for path in os.listdir(DATA_PATH):
  lines = []
  basepath = DATA_PATH + path
  filepath = basepath + '/driving_log.csv'
  with open(filepath) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      lines.append(line)
  lines = lines[1:]
  for line in lines:
    steering = line[3]

    _read_in_data(_get_path(basepath, line[0]), steering, 0)
    _read_in_data(_get_path(basepath, line[1]), steering, correction)
    _read_in_data(_get_path(basepath, line[2]), steering, -correction)

augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
  augmented_images.append(image)
  augmented_measurements.append(measurement)
  augmented_images.append(cv2.flip(image, 1))
  augmented_measurements.append(measurement * -1.0)

#X_train = np.array(augmented_images)
#y_train = np.array(augmented_measurements)
total = len(augmented_images)
def myGenerator ():
  while 1:
    for i in range(math.ceil(total / 32)):
      yield np.array(augmented_images[i* 32:(i+1)*32]), np.array(augmented_measurements[i*32:(i+1)*32])


from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Lambda

model = Sequential()
model.add(Lambda(lambda x: x / 255.0, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(myGenerator(), samples_per_epoch = total, callbacks=[], validation_data=None, nb_epoch=epochs)

#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=epochs)

model.save('model.h5')
