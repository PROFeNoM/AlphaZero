import numpy as np
from keras.models import load_model
from keras.backend import clear_session

import go
from MCTS import get_action_coord


def random_action(state):
    legal_actions = state.all_legal_moves()
    sum_legal_actions = np.sum(legal_actions)
    possibilities = legal_actions / sum_legal_actions
    num = np.random.choice(len(legal_actions), p=possibilities)  # if 9x9 -> 0 ~ 81

    if num == len(legal_actions) - 1:
        return None

    row = num // go.N
    column = num % go.N

    coord = (row, column)

    return coord


class Compare:
    def __init__(self, model):
        self.model = model

    def play(self, black_action_fn, white_action_fn):
        state: go.Position = go.Position()

        while not state.is_game_over():
            next_action = black_action_fn if state.to_play == 1 else white_action_fn
            action = next_action(state)
            state = state.play_move(action)
            print(state)
        result = state.result()
        if result == 1: return 1
        if result == 0: return 0.5
        return 0

    def compare(self):
        action_model_fn = get_action_coord(self.model)
        wins, draws, loses = 0, 0, 0
        for i in range(10):
            if i % 2 == 0:
                result = self.play(action_model_fn, random_action)
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
                result = 1 - self.play(random_action, action_model_fn)
                if result == 1:
                    print("Win")
                    wins += 1
                elif result == 0.5:
                    print("Draw")
                    draws += 1
                else:
                    print("Lose")
                    loses += 1
        print(f"{wins}-{draws}-{loses}")


if __name__ == '__main__':
    model = load_model('./model/trained.h5')
    Compare(model).compare()
    clear_session()
    del model
