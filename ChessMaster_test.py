try:
    from matplotlib import pyplot as plt
    import BoardHelper
    import DataHelper
    from ChessGlobalDefs import *
    import Plotter
    import FeatureExtractor
    import Classifiers
except ImportError:
    print("Import failed.")

a_random_file = "../dataset/train/1b1B1b2-2pK2q1-4p1rB-7k-8-8-3B4-3rb3.jpeg"


test_SysSpecs = False
test_CVFeature = False
test_DataHelper = False
test_Plotter = False
test_BoardHelper = False

test_CNN_train = False
test_CNN_predict = True

if test_CNN_train:
    cnn = Classifiers.CNNClassifier()
    train_names = DataHelper.GetFileNamesInDir(g_train_dir)
    cnn.Train(train_names)


if test_CNN_predict:
    cnn = Classifiers.CNNClassifier()
    cnn.LoadMostRecentModel()
    predicted_label = cnn.Predict(DataHelper.ReadImage(a_random_file))
    L = BoardHelper.LabelArrayToL(predicted_label)
    FEN = BoardHelper.LtoFEN(L)
    print("predicted: " + FEN)
    print("Original:  " + DataHelper.GetCleanNameByPath(a_random_file))


if test_SysSpecs:
    import os, platform
    print('OS name:', os.name, ', system:', platform.system(), ', release:', platform.release())
    import sys
    print("Anaconda version:")
    #!conda list anaconda
    print("Python version: ", sys.version)
    print("Python version info: ", sys.version_info)
    import PIL
    from PIL import Image
    print("PIL version: ", PIL.__version__)
    import matplotlib
    import matplotlib.pyplot as plt
    print("Matplotlib version: ", matplotlib.__version__)
    #import tensorflow as tf
    #print("Keras version:", tf.keras.__version__)
    import cv2
    print("OpenCV version: ", cv2.__version__)
    import numpy as np
    print("nump version: ", np.__version__)

if test_CVFeature:
    print("###### SIFT test start ######")
    import cv2
    print("Sift: decriptor size:", cv2.SIFT_create().descriptorSize())
    img = DataHelper.ReadImage(a_random_file)

    img = cv2.Canny(img,100,200)

    kp, desc = FeatureExtractor.ExtractSIFTForGrid(img, 0, 1)
    kp2, desc2 = FeatureExtractor.ExtractSIFTForGrid(img, 0, 5)
    img_kp = cv2.drawKeypoints(img, [kp, kp2], img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.subplot(2, 2, 1)
    plt.imshow(img_kp)
    plt.subplot(2, 2, 3)
    plt.bar(x = range(128), height = desc)
    plt.xticks(x = range(128))
    plt.subplot(2, 2, 4)
    plt.bar(x = range(128), height = desc2)
    plt.xticks(x = range(128))
    plt.show()
    print("###### SIFT test end ######")

    #print("###### ORB test start ######")
    #orb = cv2.ORB_create(edgeThreshold = 0) 
    #cell = BoardHelper.GetBoardCell(img, 0, 3)
    ##plt.imshow(cell)
    ##plt.show()
    #print(img.shape)
    #kp = [cv2.KeyPoint(x = 25, y = 25, _size = 10, _angle = 0)]
    #kpcv = orb.detect(img, None)
    #x = [kp.pt[0] for kp in kpcv]
    #y = [kp.pt[1] for kp in kpcv]
    #plt.imshow(img, cmap = "gray")
    #plt.scatter(x, y, color = "green")
    #plt.show()
    #kp_final, orb_feature = orb.detectAndCompute(img, None)
    #print("###### ORB test end ######")


if test_DataHelper:
    print("Test start: DataHelper")
    file_names = DataHelper.GetCleanNamesInDir(g_train_dir)
    print("got", len(file_names), "images")
    file_names = DataHelper.GetCleanNamesInDir(g_train_dir,num_return = 100)
    file_names = DataHelper.GetCleanNamesInDir(g_train_dir,num_return = 100)
    print("got", len(file_names), "images")
    full_paths = DataHelper.GetFileNamesInDir(g_train_dir, num_return=42)
    images = DataHelper.ReadImages(full_paths)
    print("got", len(images), "images of size ", images[0].shape)
    
    
    print("Test end: DataHelper")



if test_Plotter:
    print("Test start: Plotter")
    fen = DataHelper.GetCleanNameByPath(a_random_file)
    img = DataHelper.ReadImage(a_random_file)
    fig = Plotter.PlotSegmentedBoard(img, fen)
    plt.show(fig)

    print("Test end: Plotter")
    

if test_BoardHelper:
    print("Test start: BoardHelper")
    str = "1b1B1b2-2pK2q1-4p1rB-7k-8-8-3B4-3rb3"
    print("original FEN:", str)
    l = BoardHelper.FENtoL(str)
    print("FENtoL:", "".join(l))
    fen = BoardHelper.LtoFEN(l)
    print("LtoFEN:", fen)
    if (str == fen):
        print("FEN-L transform Result is correct.")
    else:
        print("FEN-L transform WRONG.")

    fen = DataHelper.GetCleanNameByPath(a_random_file)
    img = DataHelper.ReadImage(a_random_file)
    grids = BoardHelper.ImageToGrids(img, g_grid_size, g_grid_size).reshape(g_grid_num, g_grid_size, g_grid_size, 3)
    labels = BoardHelper.FENtoL(fen)
    plt.show()
    print("Test end: BoardHelper")



