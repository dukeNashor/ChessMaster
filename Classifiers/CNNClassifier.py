import Classifiers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import cv2
from skimage import io, transform
import numpy as np
import os

from ChessGlobalDefs import g_grid_size, g_image_size, g_grid_num, g_grid_row, g_grid_col, g_down_sampled_size, g_down_sampled_grid_size
import BoardHelper
import DataHelper



from matplotlib import pyplot as plt

#import tensorflow as tf
#from tensorflow import keras
#from tf.keras.models import Sequential
#from tf.keras.layers.core import Flatten, Dense, Dropout, Activation
#from tf.keras.layers.convolutional import Convolution2D

class CNNClassifier(Classifiers.IClassifier):

    # the file name format does not accept batch as parameter. link:
    # https://github.com/tensorflow/tensorflow/issues/38668
    s_check_point_file_name = "./CNN_training_checkpoint/cp_{epoch:02d}-{accuracy:.2f}.ckpt"
    s_check_point_path = os.path.dirname(s_check_point_file_name)
    s_save_frequence = 10000 # save a checkpoint every s_save_frequence batches

    def __init__(self):
        
        tf.config.threading.set_inter_op_parallelism_threads(3)
        tf.config.threading.set_intra_op_parallelism_threads(3)

        # define our model
        self.__model__ = keras.Sequential(
            [
                layers.Convolution2D(32, (3, 3), input_shape = (g_down_sampled_grid_size, g_down_sampled_grid_size, 3)),
                layers.Activation('relu'),

                layers.Convolution2D(32, (3, 3)),
                layers.Activation('relu'),
                
                layers.Convolution2D(32, (3, 3)),
                layers.Activation('relu'),

                layers.Flatten(),
                
                layers.Dense(128),
                layers.Activation('relu'),

                layers.Dropout(0.4),

                layers.Dense(13),
                layers.Activation("softmax")
            ]
        )

        self.__model__.compile(loss = "categorical_crossentropy", optimizer = 'adam', metrics = ["accuracy"])
        self.__model__.summary()

        self.__save_check_point_callback__ = tf.keras.callbacks.ModelCheckpoint(
            filepath = CNNClassifier.s_check_point_file_name,
            monitor='val_accuracy',
            save_weights_only = True,
            save_freq = CNNClassifier.s_save_frequence,
            verbose = 1
            )


    # this method should accept N * 64 * m * n numpy array as train data, and N lists of 64 chars as label.
    def Train(self, train_data_names):
        train_size = len(train_data_names)

        # generator
        def func_generator(train_file_names):
            for image_file_name in train_file_names:
                x = CNNClassifier.PreprocessImage(image_file_name)
                y = np.array(BoardHelper.FENtoOneHot(DataHelper.GetCleanNameByPath(image_file_name)))
                yield x, y
        
        # try load last checkpoint
        try:
            last_cp = tf.train.latest_checkpoint(CNNClassifier.s_check_point_path)
            self.__model__.load_weights(last_cp)
            print("Loaded checkpoint from " + last_cp)
        except:
            print("No checkpoint is loaded.")

        # train
        self.__model__.fit(func_generator(train_data_names),
                           use_multiprocessing = False,
                           #batch_size = 6400,
                           steps_per_epoch = train_size,
                           epochs = 1,
                           callbacks = [self.__save_check_point_callback__],
                           verbose = 1)


    # this should accept a 64 * m * n numpy array as query data, and returns the fen notation of the board.
    def Predict(query_data):
        pred = self.__model__.predict(query_data).argmax(axis=1).reshape(-1, 8, 8)
        print(pred)


    def SaveModel(self, save_file_name):
        path_name = os.path.dirname(save_file_name)
        os.makedirs(path_name, exist_ok=True)

        __model__.save_weights(save_file_name)

    def LoadModel(self, load_file_name):
        self.__model__.load_weights(load_file_name)

    @staticmethod
    def PreprocessImage(image_path):

        image = io.imread(image_path)
        image = transform.resize(image, (g_down_sampled_size, g_down_sampled_size), mode='constant')
        
        # 1st and 2nd dim is 8
        grids = BoardHelper.ImageToGrids(image, g_down_sampled_grid_size, g_down_sampled_grid_size)

        # debug
        #plt.imshow(grids[0][3])
        #plt.show()

        return grids.reshape(g_grid_row * g_grid_col, g_down_sampled_grid_size, g_down_sampled_grid_size, 3)