import tensorflow as tf
import numpy as np
import os
from os.path import relpath
import math

ROOT_PATH = 'C:\\Users\\Chris\\Documents\\Lip Reading in the Wild Data\\lipread_mp4'

class sequence_generator(tf.keras.utils.Sequence):
    def __init__(self, x_set, dir, batch_size):
            self.x = x_set
            self.batch_size = batch_size
            self.dir = dir
            self.label_processor = tf.keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(x_set["tag"]))

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        data = self.get_video_data(batch_x)
        return data

    def get_video_data(self, df):
        video_paths = df["video_name"].values.tolist()
        video_tags = df["tag"].values.tolist()
        labels = df["tag"].values
        labels = self.label_processor(labels[..., None]).numpy()
        data = []

        # For each video.
        for idx, path in enumerate(video_paths):
            video_path = os.path.join(ROOT_PATH, video_tags[idx])
            video_path = os.path.join(video_path, self.dir)
            video_path = os.path.join(video_path, path)
            video_path = video_path.replace('\\', '/')
            relative_path = relpath(video_path)
            frames_data = self.read_face_alignment_data_file(video_path)

            data.append(frames_data)

            # print('Loaded video: ' + video_path)

        data = np.array(data)
        return data, labels
    
    def read_face_alignment_data_file(self, path):
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