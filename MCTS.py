from cmath import sqrt

import numpy as np

from go import Position, N
from NNet import NETWORK_INPUT_SHAPE
import features

from Config import MCTS_SIMULATIONS, C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON


def get_legal_actions(moves):
    return [i for i, is_legal in enumerate(moves) if is_legal]


def flat_to_coord(flat):
    if flat == N * N:
        return None
    return flat // N, flat % N


class Node:
    def __init__(self, state: Position, model, p):
        self.state = state
        self.model = model
        self.p = p
        self.w = 0
        self.n = 0
        self.children = None

    def _evaluation(self, legal_moves):
        boardsize, _, history = NETWORK_INPUT_SHAPE
        input_features = features \
            .extract_features(self.state, features.AGZ_FEATURES) \
            .reshape(1, boardsize, boardsize, history) \
            .astype(np.float)

        y_pred = self.model.predict(input_features, batch_size=4)
        pi = y_pred[0][0][
            legal_moves]  # The fact that we take legal_moves from a parameter isn't pretty, but it is faster

        p_sum = sum(pi)
        if p_sum != 0:
            pi /= p_sum

        v = y_pred[1][0][0]

        return pi, v

    def _add_child(self, coord, p, noise):
        self.children.append(Node(
            self.state.play_move(coord),
            self.model,
            (1 - DIRICHLET_EPSILON) * p + DIRICHLET_EPSILON * noise
        ))

    def _select_child(self):
        number_of_visits = sqrt(sum([child.n for child in self.children]))
        pucb = [
            ((- child.w / child.n) if child.n else 0.0) +
            C_PUCT * child.p * number_of_visits / (1 + child.n)
            for child in self.children
        ]
        return self.children[np.argmax(pucb)]

    def search(self):
        if self.state.is_game_over():
            v = self.state.to_play * self.state.result()
            self.w += v
            self.n += 1
            return v

        if self.children is None:
            legal_moves = get_legal_actions(self.state.all_legal_moves())
            pi, v = self._evaluation(legal_moves)

            noise = np.random.dirichlet([DIRICHLET_ALPHA] * 82)

            self.w += v
            self.n += 1

            self.children = []
            for flat, p in zip(legal_moves, pi):
                if flat == N * N:
                    coord = None
                else:
                    row = flat // N
                    column = flat % N
                    coord = row, column

                self._add_child(coord, p, noise[flat])

            return v
        else:
            v = -self._select_child().search()
            self.w += v
            self.n += 1
            return v


def get_action_prob(model, state: Position, temperature):
    root: Node = Node(state, model, 0)

    for _ in range(min(MCTS_SIMULATIONS, len(get_legal_actions(state.all_legal_moves())) * 4)):
        root.search()

    scores = [child.n for child in root.children]

    if temperature == 0:
        action = np.argmax(scores)
        q = root.children[action].w / root.children[action].n
        #print("\raction : {} / legal moves: {} / scores: {} / value: {} / to_play: {}".format(action, len(scores) - 1,
        #                                                                                      scores[action], q,
        #                                                                                      state.to_play), end="")
        if state.recent and (state.recent[-1].move is None) and (state.result() == state.to_play):  # win if can win
            action = -1
            print("\tLast play is pass, win")
        #if state.to_play * q > 0.25:  # resign
        #    action = -1
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:
        xs = [x ** (1 / temperature) for x in scores]
        return [x / sum(xs) for x in xs]

    return scores


def get_action_coord(model):
    def action_fn(state):
        scores = get_action_prob(model, state, 0)
        action = np.random.choice(get_legal_actions(state.all_legal_moves()), p=scores)
        #action = get_legal_actions(state.all_legal_moves())[np.argmax(scores)]
        if action == N * N:
            return None
        return flat_to_coord(action)

    return action_fn
