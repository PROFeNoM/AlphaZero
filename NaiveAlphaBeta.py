import sys
import time
from random import choice

from typing import Dict

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

    if point_of_view == Board._BLACK:
        return black_score - white_score
    else:
        return white_score - black_score


OPENINGS = {
    "Orthodox": ["E5", "E3", "F4"],
    "Curveball": ["E5", "E3", "F4", "F7"],
    "Hand Fan": ["E5", "E3", "F4", "E7"],
    "Windmill": ["E5", "E3", "F3", "F4", "E4", "G3", "F2", "D3"],
    "Sword": ["E5", "E3", "E7", "E8"],
    "Jump Attack": ["E5", "E3", "F5", "E7", "D3"],
    "Cross Line": ["E5", "C7", "E7", "C5", "E3", "F5"],
    "Soccer Juggling": ["E5", "E3", "D7"],
    "Pendulum": ["E5", "F3", "G4"],
    "Head Butt": ["E5", "E3", "E4"],
    "New Orthodox": ["E6", "E4", "G5"],
    "Slider": ["E6", "E4", "G5", "G3"],
    "Secret Agent 033": ["E6", "E4", "G5", "G7"],
    "Andromeda": ["E6", "E4", "G5", "C5", "C6", "G4"],
    "White Slice": ["E6", "E4", "F4", "F3", "G4", "G3"],
    "Boots": ["E6", "E3", "E4", "F3", "D3"],
    "Zazen": ["E6", "E3", "E4", "F3", "F4", "D3", "D4"],
    "Kodachi": ["E6", "E4", "E3"],
    "Sea Fairy": ["E7", "E4", "E3"],
    "Lunar Eclipse": ["F4", "D6", "C7"],
    "Black Boomerang": ["F6", "D4", "E4", "E3", "D5", "C5", "E5"],
    "Bean Throwing": ["F6", "D4", "E4", "E3", "F3", "D3", "D7"],
    "Flower Fairy": ["F3", "D6", "D7"]
}


def get_book_move(board: Board) -> str:
    """
    Return the next move from the opening book.
    """
    move_history = board._historyMoveNames
    move_done = len(move_history)
    # Look in OPENINGS for a same sequence of moves.
    for opening_name, opening_moves in OPENINGS.items():
        if move_done >= len(opening_moves):
            continue
        can_use_opening = True
        for i in range(len(opening_moves)):
            if i >= move_done:
                break
            if opening_moves[i] != move_history[i]:
                can_use_opening = False
                break
        if can_use_opening:
            print("Using opening:", opening_name)
            return opening_moves[move_done]
    # If no opening found, return a random move.
    moves = board.legal_moves()
    moves.remove(-1)
    move = choice(moves)
    return board.flat_to_name(move)


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
            return self.eval_fn(board, point_of_view)
        transposition = self.transposition_table.get(str(board._currentHash))
        if transposition is not None and transposition['depth'] >= depth:
            sys.stderr.write("\r.")
            return transposition['score']
        if depth == 0 or board.is_game_over():
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
            self.transposition_table[str(board._currentHash)] = {'depth': depth, 'score': best_value}
            return best_value

    def get_action(self, board: Board, color_to_play: int):
        """
        Returns the best action for the current board using iterative deepening on alpha beta pruning.
        Move ordering is done after each depth.
        """
        self.start_time = time.monotonic()
        legal_moves = board.legal_moves()
        current_depth_evaluations: Dict[int, float] = {}  # {move: evaluation}
        depth = 1
        prev_best_move = legal_moves[0]
        while depth <= 20:
            self.transposition_table = {}
            sys.stderr.write(f"Prev: {prev_best_move}\n")
            best_move = legal_moves[0]
            best_value = -float('inf')
            sys.stderr.write(f"Depth: {depth}\n")
            # Analyze the current depth
            for move in legal_moves:
                print(f"\rAnalyzing move: {move}", end="")
                board.push(move)
                value = self.alpha_beta(board, depth, -float('inf'), float('inf'), False, color_to_play, depth != 1)
                board.pop()
                if self.is_out_of_time() and depth != 1:
                    sys.stderr.write(f"TO during search: {prev_best_move}\n")
                    return prev_best_move
                current_depth_evaluations[move] = value
            # Order legal_moves according to the new evaluations
            legal_moves = sorted(legal_moves, key=lambda m: current_depth_evaluations[m], reverse=True)
            # Update best_move and best_value
            for move in legal_moves:
                if current_depth_evaluations[move] > best_value:
                    best_move = move
                    best_value = current_depth_evaluations[move]
            sys.stderr.write(f"Best move: {best_move}\n")
            prev_best_move = best_move
            # Update depth
            depth += 1
        return prev_best_move


class myPlayer(PlayerInterface):
    def __init__(self):
        self.state: Board = Board()
        self.player_early = IterativeDeepeningAlphaBeta(time_per_move=7, eval_fn=eval_fn)
        self.player_mid = IterativeDeepeningAlphaBeta(time_per_move=20, eval_fn=eval_fn)
        self.player_late = IterativeDeepeningAlphaBeta(time_per_move=7, eval_fn=eval_fn)
        self.color = None
        self.turn = 0

    def getPlayerName(self):
        return "Dans La Légende"

    def getPlayerMove(self):
        self.turn += 1
        if self.state.is_game_over():
            return "PASS"

        if self.turn < 20:
            move = Board.name_to_flat(get_book_move(self.state))
        elif self.turn < 60:
            move = self.player_early.get_action(self.state, self.color)
        elif self.turn < 120:
            move = self.player_mid.get_action(self.state, self.color)
        else:
            move = self.player_late.get_action(self.state, self.color)

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
        pass


if __name__ == "__main__":
    player = myPlayer()
    player.newGame(Board._BLACK)
    player.getPlayerMove()
