import cv2
import numpy as np
import glob
import os

# global vars
g_dataset_dir = "../dataset/"
g_train_dir = g_dataset_dir + "/train/"
g_test_dir = g_dataset_dir + "/test/"



# class to hold data io operations
class DataHelper:

    # get clean name by a path, where in our case this gets the FEN conviniently
    @staticmethod
    def GetCleanNameByPath(file_name):
        return os.path.splitext(os.path.basename(file_name))[0]

    # get full paths to the files in a directory.
    @staticmethod
    def GetFileNamesInDir(path_name, extension="*", num_return = 0):
        if num_return == 0:
            return glob.glob(path_name + "/*." + extension)
        else:
            return glob.glob(path_name + "/*." + extension)[:num_return]

    # get name list
    @staticmethod
    def GetCleanNamesInDir(path_name, extension = "*", num_return = 0):
        names = DataHelper.GetFileNamesInDir(path_name, extension)
        offset = len(extension) + 1
        clean_names = [os.path.basename(x)[:-offset] for x in names]
        if num_return == 0:
            return clean_names
        else:
            return clean_names[:num_return]

    # read dataset
    @staticmethod
    def ReadImages(file_names, path = ""):
        if path == "":
            return [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in file_names]
        else:
            return [cv2.imread(path + "/" + f, cv2.IMREAD_GRAYSCALE) for f in file_names]

    # read image by name
    @staticmethod
    def ReadImage(file_name, format = cv2.IMREAD_GRAYSCALE):
        return cv2.imread(file_name, format)


