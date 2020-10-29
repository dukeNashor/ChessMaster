from BoardHelper import *
from DataHelper import *
from ChessGlobalDefs import *
from Plotter import *

a_random_file = "../dataset/train/1b1B1b2-2pK2q1-4p1rB-7k-8-8-3B4-3rb3.jpeg"


test_DataHelper = False
test_Plotter = False
test_BoardHelper = True



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
    fen = DataHelper.GetCleanNameByPath(a_random_file)
    img = DataHelper.ReadImage(a_random_file)
    index2, y_1 = BoardHelper.ReadBoardFEN(img, fen)
    plt.imshow(index2[1].reshape(50, 50), cmap = plt.cm.gray)
    plt.show()
    print("Test end: BoardHelper")
