import pickle
from datetime import datetime

import numpy as np
from keras.models import load_model
from tqdm import tqdm

import features
from go import Position
from MCTS import get_action_prob, get_legal_actions, flat_to_coord
from NNet import NETWORK_OUTPUT_SIZE
from keras.backend import clear_session

SELFPLAY_GAMES = 10
SELFPLAY_TEMPERATURE_THRESHOLD = 15
SELFPLAY_TEMPERATURE_EARLY = 1
SELFPLAY_TEMPERATURE_TERMINAL = 0


class SelfPlay:
    def __init__(self):
        self.model = None
        self.history = []

    def _play_game(self):
        game_history = []
        state: Position = Position()

        number_of_moves = 0
        while not state.is_game_over():
            if number_of_moves < SELFPLAY_TEMPERATURE_THRESHOLD:
                scores = get_action_prob(self.model, state, SELFPLAY_TEMPERATURE_EARLY)
            else:
                scores = get_action_prob(self.model, state, SELFPLAY_TEMPERATURE_TERMINAL)

            pi = [0] * NETWORK_OUTPUT_SIZE
            for a, p in zip(get_legal_actions(state.all_legal_moves()), scores):
                pi[a] = p

            input_features = features.extract_features(state, features.AGZ_FEATURES)

            game_history.append([input_features, pi, None])

            flat = get_legal_actions(state.all_legal_moves())[np.argmax(scores)]
            #print(f"Move chose {flat}, i.e., {flat_to_coord(flat)}")
            state = state.play_move(flat_to_coord(flat))
            #print(state)
            number_of_moves += 1
        #print(f"Number of moves: {number_of_moves}")
        v = state.result()
        for i in range(len(game_history)):
            game_history[i][2] = v
            v *= -1

        return game_history

    def _save_history(self):
        path = f"./data/{datetime.now().timestamp()}.npy"
        with open(path, mode='wb') as f:
            pickle.dump(self.history, f)

    def generate_data(self):
        self.history = []
        self.model = load_model('./model/best.h5')
        for _ in tqdm(range(SELFPLAY_GAMES), desc="Self play"):
            samples = self._play_game()
            self.history.extend(samples)

        self._save_history()
        clear_session()
        del self.model

        return self.history

