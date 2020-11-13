import Classifiers

from sklearn import svm

from ChessGlobalDefs import *
import BoardHelper
import DataHelper

import numpy as np

from skimage import io, transform
import skimage.util
from skimage.util.shape import view_as_blocks

from matplotlib import pyplot as plt

# interface of the classifiers
class SVCClassifier(Classifiers.IClassifier):

    def __init__(self):
        self.__svc__ = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                          decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
                          max_iter=-1, probability=False, random_state=None, shrinking=True,
                          tol=0.001, verbose=True)

    # this method should accept a list of file names of the training data
    def Train(self, train_file_names):
        num_files = len(train_file_names)
        xs = np.empty([num_files * g_grid_num, g_down_sampled_grid_size * g_down_sampled_grid_size])

        i = 0

        ys = np.empty(shape=(num_files * g_grid_num))
        for f in train_file_names:
            img = io.imread(f, as_gray = True)
            grids = SVCClassifier.SVCPreprocess(img)
            # x
            xs[i:i + g_grid_num, :] = grids
            # y
            y = np.array(BoardHelper.FENtoOneHot(DataHelper.GetCleanNameByPath(f))).argmax(axis=1)
            ys[i : i + g_grid_num] = y
            i += g_grid_num


        # train
        print("start training.")
        self.__svc__.fit(xs, ys)

        print("finished.")

    # this should accept a 400 * 400 * 3 numpy array as query data, and returns the fen notation of the board.
    def Predict(self, query_data):
        grids = SVCClassifier.SVCPreprocess(query_data)
        y_pred = self.__svc__.predict(grids)
        
        return BoardHelper.LabelArrayToL(y_pred)

    @staticmethod
    def SVCPreprocess(img):
        img = transform.resize(img, (g_down_sampled_size, g_down_sampled_size), mode='constant')
        grids = skimage.util.shape.view_as_blocks(img, block_shape = (g_down_sampled_grid_size, g_down_sampled_grid_size))
        grids = grids.reshape((-1, grids.shape[3], grids.shape[3]))
        grids = grids.reshape((grids.shape[0], grids.shape[1] * grids.shape[1]))
        return grids