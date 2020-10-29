import re
import string
from collections import OrderedDict 

import numpy as np


# helper functions as static class methods.
class BoardHelper:

    #FEN TO LABELS OF SQUARES
    @staticmethod
    def FENtoL(fen): 
        rules = {
            r"-": r"",
            r"1": r"0",
            r"2": r"00",
            r"3": r"000",
            r"4": r"0000",
            r"5": r"00000",
            r"6": r"000000",
            r"7": r"0000000",
            r"8": r"00000000",
        }
        if type(fen) == str:
            fen = [fen]

        for s in fen:
            for key in rules.keys():
                s = re.sub(key, rules[key], s)
            return list(s)

    # get 8*8 char matrix
    @staticmethod
    def LtoCharMat(l):
        if type(l) == list:
            return np.array(l).reshape((8,8))
        if type(l) == str:
            return np.array([l]).reshape((8,8))


    # function to split into 64 square (modified from https://www.kaggle.com/yeahlan/chess-positions-fen-generator)
    # accepts grayscale image as input e.g. img = cv2.imread('../dataset/'+location+'/'+name+'.jpeg',cv2.IMREAD_GRAYSCALE)
    @staticmethod
    def ReadBoardFEN(board_image, fen):
        y_1 = BoardHelper.FENtoL(fen)

        # Divide the picture into 64 pieces
        # TODO: removed the hard coded size
        size=50 
        index2=np.zeros((64,size**2))
        for i in range(8):
            for j in range(8):
                index2[i*8+j,:]=np.array(board_image)[i*size:(i+1)*size,j*size:(j+1)*size].reshape(1,size**2)

        return index2, y_1

    # Overload of the above function.
    @staticmethod
    def ReadBoardString(board_image, name_str):
        return ReadBoardFEN(board_image, FENtoL(name_str))
    



