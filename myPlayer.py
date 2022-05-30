from NaiveAlphaBeta import myPlayer as AlphaBetaPlayer


class myPlayer(AlphaBetaPlayer):
    def __init__(self):
        super().__init__()
        self.name = "myPlayer"

    def getPlayerName(self):
        return self.name

    def getPlayerMove(self):
        return super().getPlayerMove()

    def playOpponentMove(self, move):
        super().playOpponentMove(move)
