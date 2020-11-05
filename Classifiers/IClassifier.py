import abc

# A good reference of Python OOP:
# https://www.cs.colorado.edu/~kena/classes/5448/f12/presentation-materials/li.pdf

# interface of the classifiers
class IClassifier:

    # this method should accept N * 64 * m * n numpy array as train data, and N lists of 64 chars as label.
    @abc.abstractmethod
    def Train(self, train_file_names):
        raise NotImplementedError()

    # this should accept a 64 * m * n numpy array as query data, and returns the fen notation of the board.
    @abc.abstractmethod
    def Predict(self, query_data):
        raise NotImplementedError()

        