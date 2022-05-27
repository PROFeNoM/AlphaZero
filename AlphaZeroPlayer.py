# -*- coding: utf-8 -*-
from keras.models import load_model

from MCTS import get_action_coord
from playerInterface import *
from go import Position, N

indexLetters = {letter: index for index, letter in enumerate('ABCDEFGHJKLMNOPQRST'[:N])}


def coord_to_name(coord):
    if coord is None:
        return 'PASS'
    row = coord[0]
    column = coord[1]
    return 'ABCDEFGHJKLMNOPQRST'[column] + str(N - row)


def name_to_coord(name):
    if name == 'PASS':
        return None
    col = indexLetters[name[0]]
    row = N - int(name[1:])
    return row, col


class myPlayer(PlayerInterface):
    ''' AlphaZero based agent. '''

    def __init__(self):
        self.state: Position = Position()
        self.model = load_model('./model/best.h5')
        self.action_model_fn = get_action_coord(self.model)
        self.color = None

    def getPlayerName(self):
        return "Dans La Légende"

    def getPlayerMove(self):
        if self.state.is_game_over():
            print("Referee told me to play but the game is over!")
            return "PASS"

        next_action = self.action_model_fn(self.state)
        self.state = self.state.play_move(next_action)
        print("Dans La Légende played ", coord_to_name(next_action), "i.e. ", next_action)
        return coord_to_name(next_action)

    def playOpponentMove(self, move):
        self.state = self.state.play_move(name_to_coord(move))

    def newGame(self, color):
        self.state = Position()
        self.color = color

    def endGame(self, color):
        if self.color == color:
            print("Usual work.")
        else:
            print("Idc.")
