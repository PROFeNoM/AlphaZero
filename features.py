# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Features used by AlphaGo Zero, in approximate order of importance.
Feature                 # Notes
Stone History           16 The stones of each color during the last 8 moves.
Ones                    1  Constant plane of 1s
All features with 8 planes are 1-hot encoded, with plane i marked with 1
only if the feature was equal to i. Any features >= 8 would be marked as 8.
This file includes the features from the first paper as DEFAULT_FEATURES
and the features from AGZ as AGZ_FEATURES.
"""

import numpy as np
import go

# Resolution/truncation limit for one-hot features
P = 8


def planes(num_planes):
    def deco(f):
        f.planes = num_planes
        return f

    return deco


# TODO(tommadams): add a generic stone_features for all N <= 8
@planes(16)
def stone_features(position):
    # a bit easier to calculate it with axis 0 being the 16 board states,
    # and then roll axis 0 to the end.
    features = np.zeros([16, go.N, go.N], dtype=np.uint8)

    num_deltas_avail = position.board_deltas.shape[0]
    cumulative_deltas = np.cumsum(position.board_deltas, axis=0)
    last_eight = np.tile(position.board, [8, 1, 1])
    # apply deltas to compute previous board states
    last_eight[1:num_deltas_avail + 1] -= cumulative_deltas
    # if no more deltas are available, just repeat oldest board.
    last_eight[num_deltas_avail +
               1:] = last_eight[num_deltas_avail].reshape(1, go.N, go.N)

    features[::2] = last_eight == position.to_play
    features[1::2] = last_eight == -position.to_play
    return np.rollaxis(features, 0, 3)


# TODO(tommadams): add a generic stone_features for all N <= 8
@planes(8)
def stone_features_4(position):
    # a bit easier to calculate it with axis 0 being the 16 board states,
    # and then roll axis 0 to the end.
    features = np.zeros([8, go.N, go.N], dtype=np.uint8)

    num_deltas_avail = position.board_deltas.shape[0]
    # print(num_deltas_avail)
    # print('aaaa')
    cumulative_deltas = np.cumsum(position.board_deltas, axis=0)
    last = np.tile(position.board, [4, 1, 1])
    # print(cumulative_deltas.shape)
    # apply deltas to compute previous board states
    last[1:num_deltas_avail + 1] -= cumulative_deltas
    # if no more deltas are available, just repeat oldest board.
    last[num_deltas_avail + 1:] = last[num_deltas_avail].reshape(1, go.N, go.N)

    features[::2] = last == position.to_play
    features[1::2] = last == -position.to_play
    # print('---------------------')
    # print(np.rollaxis(features, 0, 3).shape)
    return np.rollaxis(features, 0, 3)


@planes(1)
def color_to_play_feature(position):
    if position.to_play == go.BLACK:
        return np.ones([go.N, go.N, 1], dtype=np.uint8)
    else:
        return np.zeros([go.N, go.N, 1], dtype=np.uint8)


AGZ_FEATURES = [
    stone_features_4,
    color_to_play_feature
]

AGZ_FEATURES_PLANES = sum(f.planes for f in AGZ_FEATURES)


def extract_features(position, features):
    return np.concatenate([feature(position) for feature in features], axis=2)
