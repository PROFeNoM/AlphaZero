from tqdm import tqdm

import go
from MCTS import get_action_coord
from keras.models import load_model
from keras.backend import clear_session

from Config import ARENA_GAME_COUNT


def _play_game(black_action_fn, white_action_fn):
    state = go.Position()

    while not state.is_game_over():
        player_action_fn = black_action_fn if state.to_play == 1 else white_action_fn
        state = state.play_move(player_action_fn(state))

    result = state.result()
    print('Game Over', result)

    if result == 1:
        return 1
    elif result == 0:
        return 1e-4
    else:
        return 0


class Arena:
    def __init__(self):
        self.best_model_action_fn = None
        self.latest_model_action_fn = None
        self.best_model = None
        self.latest_model = None

    def play_games(self):
        self.latest_model = load_model('./model/latest.h5')
        self.best_model = load_model('./model/best.h5')

        self.latest_model_action_fn = get_action_coord(self.latest_model)
        self.best_model_action_fn = get_action_coord(self.best_model)

        wins = 0
        draws = 0
        loses = 0

        for i in tqdm(range(ARENA_GAME_COUNT), desc="Arena"):
            if i % 2 == 0:
                result = _play_game(self.latest_model_action_fn, self.best_model_action_fn)
                if result == 1:
                    print("Win")
                    wins += 1
                elif result == 0.5:
                    print("Draw")
                    draws += 1
                else:
                    print("Lose")
                    loses += 1
            else:
                result = 1 - _play_game(self.best_model_action_fn, self.latest_model_action_fn)
                if result == 1:
                    print("Win")
                    wins += 1
                elif result == 0.5:
                    print("Draw")
                    draws += 1
                else:
                    print("Lose")
                    loses += 1

        print(f"Evaluation result {wins}-{draws}-{loses}")

        clear_session()
        del self.latest_model
        del self.best_model

        return wins, draws, loses


if __name__ == '__main__':
    Arena().play_games()
