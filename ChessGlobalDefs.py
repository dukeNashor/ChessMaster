
# global instance of mapping of char vs chess pieces
# reference: Forsyth–Edwards Notation, https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
# 
# pawn = "P", knight = "N", bishop = "B", rook = "R", queen = "Q" and king = "K"
# White pieces are designated using upper-case letters ("PNBRQK") while black pieces use lowercase ("pnbrqk")

g_piece_mapping = {
    "P" : "pawn",
    "N" : "knight",
    "B" : "bishop",
    "R" : "rook",
    "Q" : "queen",
    "K" : "king",

    "p" : "pawn",
    "n" : "knight",
    "b" : "bishop",
    "r" : "rook",
    "q" : "queen",
    "k" : "king"
}
