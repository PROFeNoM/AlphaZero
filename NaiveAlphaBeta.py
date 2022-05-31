import time
import gzip
import json
from typing import Dict

from random import shuffle
from Goban import Board
from playerInterface import PlayerInterface


def eval_fn(board: Board, point_of_view: int) -> float:
    """ Naive evaluation function, comparing scores of both players. """
    black_score, white_score = board.compute_score()
    if board.is_game_over():
        if black_score > white_score:
            winner = Board._BLACK
        elif black_score < white_score:
            winner = Board._WHITE
        else:
            winner = Board._EMPTY

        if winner == point_of_view:
            return float('inf')
        else:
            return float('-inf')

    score_feature = black_score - white_score if point_of_view == Board._BLACK else white_score - black_score

    liberty_feature = 0
    ennemy_color = Board._BLACK if point_of_view == Board._WHITE else Board._WHITE
    for fcoord in range(len(board)):  # makes use of __len__
        cell = board[fcoord]  # makes use of __getitem__
        if cell == ennemy_color:
            liberty_feature -= board._stringLiberties[board._getStringOfStone(fcoord)]
        elif cell == point_of_view:
            liberty_feature += board._stringLiberties[board._getStringOfStone(fcoord)]

    return score_feature * 10 + liberty_feature


POSITION_SCORE = [0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 1, 1, 1, 1, 1, 1, 1, 0,
                  0, 1, 2, 2, 1, 2, 2, 1, 0,
                  0, 1, 2, 1, 1, 1, 2, 1, 0,
                  0, 1, 1, 1, 1, 1, 1, 1, 0,
                  0, 1, 2, 1, 1, 1, 2, 1, 0,
                  0, 1, 2, 2, 1, 2, 2, 1, 0,
                  0, 1, 1, 1, 1, 1, 1, 1, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0]


def evaluate(board: Board, point_of_view: int) -> float:
    """ Multivariate evaluation function. """
    black_score, white_score = board.compute_score()
    if board.is_game_over():
        if black_score > white_score:
            winner = Board._BLACK
        elif black_score < white_score:
            winner = Board._WHITE
        else:
            winner = Board._EMPTY

        if winner == point_of_view:
            return float('inf')
        else:
            return float('-inf')

    score_feature = black_score - white_score if point_of_view == Board._BLACK else white_score - black_score

    liberty_feature = 0
    position_feature = 0
    ennemy_color = Board._BLACK if point_of_view == Board._WHITE else Board._WHITE
    for fcoord in range(len(board)):  # makes use of __len__
        cell = board[fcoord]  # makes use of __getitem__
        if cell == ennemy_color:
            liberty_feature -= board._stringLiberties[board._getStringOfStone(fcoord)]
            position_feature -= POSITION_SCORE[fcoord]
        elif cell == point_of_view:
            liberty_feature += board._stringLiberties[board._getStringOfStone(fcoord)]
            position_feature += POSITION_SCORE[fcoord]

    return score_feature * 5 + liberty_feature * 2 + position_feature


with gzip.open("samples-9x9.json.gz") as fz:
    tables = json.loads(fz.read().decode("utf-8"))


def search_in_tables(move_history):
    len_history = len(move_history)
    next_to_play = "black" if len_history % 2 == 0 else "white"
    best_table = None
    best_wr = -1
    for table in tables:
        if len_history >= table['depth']:
            continue
        if move_history == table['list_of_moves'][:len_history] \
                and table[next_to_play + "_wins"] / table['rollouts'] > best_wr:
            best_table = table
            best_wr = table[next_to_play + "_wins"] / table['rollouts']

    return best_table['list_of_moves'][len_history] if best_table else None


def get_book_move(board: Board) -> str:
    """
    Return the next move from the opening book.
    """
    move_history = board._historyMoveNames
    move = search_in_tables(move_history)
    return move


class IterativeDeepeningAlphaBeta:
    """
    Iterative Deepening Alpha Beta with move ordering after each depth
    """

    def __init__(self, time_per_move: float, eval_fn):
        self.time_per_move = time_per_move
        self.eval_fn = eval_fn
        self.start_time = 0
        self.transposition_table = {}  # hash -> {depth: int, score: float}

    def is_out_of_time(self):
        return time.monotonic() - self.start_time > self.time_per_move

    def alpha_beta(self, board: Board, depth: int, alpha: float, beta: float, maximizing_player: bool,
                   point_of_view: int, activate_timer: bool = True) -> float:
        if activate_timer and self.is_out_of_time():
            return 0  # The return value is not used anyway in this case.

        transposition = self.transposition_table.get(str(board._currentHash))
        if transposition is not None and transposition['depth'] >= depth:
            # Use the transposition table to speed up the search.
            return transposition['score']

        if depth == 0 or board.is_game_over():
            # We are at the leaf of the tree.
            utility = self.eval_fn(board, point_of_view)
            self.transposition_table[str(board._currentHash)] = {'depth': depth, 'score': utility}
            return utility

        if maximizing_player:
            best_value = -float('inf')
            for move in board.legal_moves():
                board.push(move)
                value = self.alpha_beta(board, depth - 1, alpha, beta, False, point_of_view, activate_timer)
                board.pop()
                best_value = max(best_value, value)
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break
            # Store the best value in the transposition table.
            self.transposition_table[str(board._currentHash)] = {'depth': depth, 'score': best_value}
            return best_value
        else:
            best_value = float('inf')
            for move in board.legal_moves():
                board.push(move)
                value = self.alpha_beta(board, depth - 1, alpha, beta, True, point_of_view, activate_timer)
                board.pop()
                best_value = min(best_value, value)
                beta = min(beta, best_value)
                if beta <= alpha:
                    break
            # Store the best value in the transposition table.
            self.transposition_table[str(board._currentHash)] = {'depth': depth, 'score': best_value}
            return best_value

    def get_action(self, board: Board, color_to_play: int, max_depth: int = 20, activate_timer: bool = True) -> int:
        """
        Returns the best action for the current board using iterative deepening on alpha beta pruning.
        Move ordering is done after each depth.
        """
        self.start_time = time.monotonic()

        legal_moves = board.legal_moves()
        shuffle(legal_moves)  # Randomize the order of the moves, so that the moves aren't always the same.
        current_depth_evaluations: Dict[int, float] = {}  # {move: evaluation}
        depth = 0
        prev_best_move = legal_moves[0]

        while depth <= max_depth:  # Iterative deepening.
            self.transposition_table = {}
            best_move = legal_moves[0]
            best_value = -float('inf')

            # Analyze the current depth
            for move in legal_moves:
                board.push(move)
                value = self.alpha_beta(board, depth, -float('inf'), float('inf'), False, color_to_play,
                                        activate_timer and depth != 1)
                board.pop()
                if self.is_out_of_time() and depth != 1:
                    print(f"Played until depth {depth - 1}")
                    return prev_best_move
                current_depth_evaluations[move] = value

            # Order legal_moves according to the new evaluations
            legal_moves = sorted(legal_moves, key=lambda m: current_depth_evaluations[m], reverse=True)

            # Update best_move and best_value
            for move in legal_moves:
                if current_depth_evaluations[move] > best_value:
                    best_move = move
                    best_value = current_depth_evaluations[move]
            prev_best_move = best_move

            # Update depth
            depth += 1
        print(f"Played to depth {depth - 1}")
        return prev_best_move


class myPlayer(PlayerInterface):
    def __init__(self):
        self.state: Board = Board()
        self.player_fallback = IterativeDeepeningAlphaBeta(time_per_move=1000,
                                                           eval_fn=evaluate)  # For this player, time won't matter anyway, as we won't activate it.
        self.player_early = IterativeDeepeningAlphaBeta(time_per_move=7, eval_fn=eval_fn)
        self.player_mid = IterativeDeepeningAlphaBeta(time_per_move=20, eval_fn=eval_fn)
        self.color = None
        self.turn = 0
        self.activate_book = True
        self.time_left = 15 * 60

    def getPlayerName(self):
        return "Dans La Légende"

    def _getMove(self, state: Board, color: int, player: IterativeDeepeningAlphaBeta,
                 max_depth: int = 20, activate_timer: bool = True) -> int:
        """
        Returns the best move for the current board using iterative deepening on alpha beta pruning.
        Move ordering is done after each depth.
        """
        if self.activate_book:
            move = get_book_move(state)
        else:
            move = None

        if move is None:
            self.activate_book = False  # Don't use the book in the following turns.
            return player.get_action(state, color, max_depth=max_depth, activate_timer=activate_timer)
        else:
            print("Using book move.")
            return Board.name_to_flat(move)

    def getPlayerMove(self):
        self.turn += 1
        if self.state.is_game_over():
            return "PASS"

        start_time = time.monotonic()
        if self.time_left < 5:
            move = self._getMove(self.state, self.color, self.player_fallback, max_depth=0, activate_timer=False)
        elif self.turn < 20:
            move = self._getMove(self.state, self.color, self.player_fallback, max_depth=0, activate_timer=False)
        elif self.turn < 50:
            move = self._getMove(self.state, self.color, self.player_early)
        else:
            move = self._getMove(self.state, self.color, self.player_mid)

        end_time = time.monotonic()
        self.time_left -= (end_time - start_time)
        print("Time left:", self.time_left)
        self.state.push(move)
        return Board.flat_to_name(move)

    def playOpponentMove(self, move):
        self.turn += 1
        self.state.push(Board.name_to_flat(move))

    def newGame(self, color):
        self.color = color
        self.state = Board()
        self.turn = 0

    def endGame(self, winner):
        if winner == self.color:
            print("Plus je me rapproche du sommet, plus j’entends le ciel qui gronde.")
        else:
            print("Recherche du bonheur, j’m’enfonce dans le vide.")


if __name__ == "__main__":
    player = myPlayer()
    player.newGame(Board._BLACK)
    player.getPlayerMove()
