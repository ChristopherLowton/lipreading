import math
from tensorflow import keras
from imutils import paths
from os.path import relpath
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Reshape, Conv2D, MaxPooling2D, Convolution3D, MaxPooling3D, LSTM, BatchNormalization, LeakyReLU, Embedding, Masking
from tensorflow.keras.callbacks import EarlyStopping

from sequence_generator import sequence_generator

# print(device_lib.list_local_devices())

ROOT_PATH = 'C:\\Users\\Chris\\Documents\\Lip Reading in the Wild Data\\lipread_mp4'

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

def read_face_alignment_data_file(path):
    data = []
    f = open(path, "r")
    for line in f:
        data_points = line.replace("],[", "] [").replace("\n", "").split(" ")
        for i, point in enumerate(data_points):
            data_points[i] = point.replace('[', '').replace(']','').split(',')
            splice_coords = []
            for coord_index, coord in enumerate(data_points[i]):
                #If empty coordinate splice from data as it will be a false value due to errors in the face alignment program
                if not coord or coord.isspace():
                    splice_coords.append(coord_index)
            #Remove invalid coordinates from data in reverse order so indexes aren't compromised
            for coord_index in sorted(splice_coords, reverse=True):
                data_points[i].pop(coord_index)
            data_points[i][0] = float(data_points[i][0])
            data_points[i][1] = float(data_points[i][1])
        data.append(data_points)
    #print(str(data))
    return data

label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(train_df["tag"]))
vocabulary = label_processor.get_vocabulary()
print(str(vocabulary))

no_classes = len(vocabulary)

def prepare_all_videos(df, dir):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    video_tags = df["tag"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()
    data = []

    # For each video.
    for idx, path in enumerate(video_paths):
        video_path = os.path.join(ROOT_PATH, video_tags[idx])
        video_path = os.path.join(video_path, dir)
        video_path = os.path.join(video_path, path)
        video_path = video_path.replace('\\', '/')
        relative_path = relpath(video_path)
        frames_data = read_face_alignment_data_file(video_path)

        data.append(frames_data)

        # print('Loaded video: ' + video_path)

    data = np.array(data)
    return data, labels

def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.show()

model = tf.keras.Sequential([
    #Layer 1
    Masking(mask_value=0.0, input_shape=(29, 12, 2)),
    Conv2D(filters = 144, kernel_size=(3, 3), activation='relu', input_shape=(29, 12, 2)),
    BatchNormalization(),
    LeakyReLU(),
    MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format='channels_last'),
    Dropout(.25),

    #Layer 2
    Masking(mask_value=0.0, input_shape=(29, 12, 2)),
    Conv2D(filters = 144, kernel_size=(3, 3), activation='relu', input_shape=(29, 12, 2)),
    BatchNormalization(),
    LeakyReLU(),
    MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format='channels_last'),
    Dropout(.25),

    #Dense layer
    Flatten(input_shape=(29, 12, 2)),
    Dropout(.25),
    Dense(144, activation='relu'),
    Dense(no_classes, activation='softmax')
])

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

print(str(model.summary()))

batch_size = 50
steps_per_epoch = math.ceil(len(train_df) / batch_size)

#Shuffle data
train_df = train_df.sample(frac=1).reset_index(drop=True)
val_df = val_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)

train_generator = sequence_generator(train_df, "train", batch_size)
val_generator = sequence_generator(val_df, "val", batch_size)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)


history = model.fit(train_generator, epochs=500, validation_data=val_generator, steps_per_epoch=steps_per_epoch, callbacks=[early_stopping], shuffle=True)

# print("----- Loading video data -----")
test_data, test_labels = prepare_all_videos(test_df, "test")
# print("----- Video data loaded -----")

test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

prediction = model.predict(test_data[0:1])
prediction_index = prediction.argmax(axis=-1)[0]
prediction_label =  vocabulary[prediction_index]

print("Actual: " + str(vocabulary[test_labels[0][0]]))
print("Prediction: " + prediction_label)

prediction = model.predict(test_data[1:2])
prediction_index = prediction.argmax(axis=-1)[0]
prediction_label =  vocabulary[prediction_index]

print("Actual: " + str(vocabulary[test_labels[1][0]]))
print("Prediction: " + prediction_label)

prediction = model.predict(test_data[2:3])
prediction_index = prediction.argmax(axis=-1)[0]
prediction_label =  vocabulary[prediction_index]

print("Actual: " + str(vocabulary[test_labels[2][0]]))
print("Prediction: " + prediction_label)

prediction = model.predict(test_data[3:4])
prediction_index = prediction.argmax(axis=-1)[0]
prediction_label =  vocabulary[prediction_index]

print("Actual: " + str(vocabulary[test_labels[3][0]]))
print("Prediction: " + prediction_label)

prediction = model.predict(test_data[4:5])
prediction_index = prediction.argmax(axis=-1)[0]
prediction_label =  vocabulary[prediction_index]

print("Actual: " + str(vocabulary[test_labels[4][0]]))
print("Prediction: " + prediction_label)


plot_metric(history, 'loss')
plot_metric(history, 'accuracy')