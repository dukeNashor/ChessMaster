import matplotlib.pyplot as plt

from BoardHelper import *


# class to facilitate plots of images and board areas. Mainly used for debugging.
class Plotter:

    def PlotSegmentedBoard(board_image, fen):
        # read indices and get l
        index2, y_1 = BoardHelper.ReadBoardFEN(board_image, fen)

        fig = plt.figure(figsize=(15, 15))
        plt.suptitle(fen, size=16)
        for i in range(64):
            plt.subplot(8, 8, i+1)
            plt.imshow(index2[i,:].reshape(50,50),cmap=plt.cm.gray)
            plt.title(np.array(y_1).reshape(-1)[i])
            plt.xticks(())
            plt.yticks(())
        
        return fig
