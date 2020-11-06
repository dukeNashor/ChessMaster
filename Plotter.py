from ChessGlobalDefs import *
import BoardHelper

import numpy as np
import matplotlib.pyplot as plt

# class to facilitate plots of images and board areas. Mainly used for debugging.
def PlotSegmentedBoard(board_image, fen):

    grids = BoardHelper.ImageToGrids(board_image, g_grid_size, g_grid_size).reshape(g_grid_num, g_grid_size, g_grid_size, 3)
    labels = BoardHelper.FENtoL(fen)

    fig = plt.figure(figsize=(15, 15))
    plt.suptitle(fen, size=16)
    for i in range(64):
        plt.subplot(8, 8, i+1)
        plt.imshow(grids[i,:])
        plt.title(labels[i])
        plt.xticks(())
        plt.yticks(())
        
    return fig
