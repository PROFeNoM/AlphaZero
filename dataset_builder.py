import gzip
import json
import os.path
import pickle
import urllib.request
from pathlib import Path

import numpy as np
from tqdm import tqdm

import features
import go
from GnuGo import *


def write_samples(hist):
    cur_dir = Path(__file__).parent.absolute()
    cur_dir = cur_dir / 'data'
    os.makedirs(cur_dir, exist_ok=True)
    path = str(cur_dir) + '\\samples.history'

    with open(path, mode='wb') as f:
        pickle.dump(hist, f)


def get_raw_data_go():
    raw_samples_file = "samples-9x9.json.gz"

    if not os.path.isfile(raw_samples_file):
        print(f"File {raw_samples_file} not found. Downloading...")
        urllib.request.urlretrieve("https://www.labri.fr/perso/lsimon/ia-inge2/samples-9x9.json.gz", raw_samples_file)
        print("Download complete.")

    with gzip.open(raw_samples_file) as fz:
        data = json.loads(fz.read().decode("utf-8"))
    return data


indexLetters = {letter: index for index, letter in enumerate('ABCDEFGHJKLMNOPQRST'[:go.N])}


def name_to_coord(name):
    if name == 'PASS':
        return None
    col = indexLetters[name[0]]
    row = int(name[1:]) - 1
    return row, col


def coord_to_flat(coord):
    return coord[0] * go.N + coord[1]


def flat_to_coord(flat):
    return flat // go.N, flat % go.N


def name_to_flat(name):
    return coord_to_flat(name_to_coord(name))


def get_prob_reward(table, gnugo: GnuGo):
    moves = gnugo.Moves(gnugo)

    for move in table['list_of_moves']:
        moves.playthis(move)

    status, _ = moves._gnugo.query("experimental_score " + moves._nextplayer)

    if status != "OK":
        return None

    status, possible_moves = moves._gnugo.query("top_moves " + moves._nextplayer)

    possible_moves = possible_moves.strip().split()

    if len(possible_moves) == 0:
        return None, None

    best_moves = [m for idx, m in enumerate(possible_moves) if idx % 2 == 0]
    scores = np.array([float(s) for idx, s in enumerate(possible_moves) if idx % 2 == 1])

    assert len(best_moves) == len(scores)

    prob_distr = scores / scores.sum()
    probs = [0] * (go.N * go.N + 1)

    for idx, m in enumerate(best_moves):
        flat_move = name_to_flat(m)
        probs[flat_move] = prob_distr[idx]

    if table['depth'] % 2 == 0:  # black plays next
        reward = table['black_wins'] / table['rollouts']
    else:  # white plays next
        reward = table['white_wins'] / table['rollouts']

    gnugo.query("clear_board")

    return probs, reward


if __name__ == '__main__':
    gnugo = GnuGo(9)

    tables = get_raw_data_go()

    samples = []

    for i, table in enumerate(tqdm(tables)):
        history = []

        state: go.Position = go.Position()

        for move in table['list_of_moves']:
            state.play_move(name_to_coord(move))

        x = features.extract_features(state, features.AGZ_FEATURES)
        p, r = get_prob_reward(table, gnugo)
        if x is not None and p is not None and r is not None:
            history.append([x, p, r])
            samples.extend(history)

    write_samples(samples)
