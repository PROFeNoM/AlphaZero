from shutil import copy
from tqdm import tqdm

from Arena import Arena
from SelfPlay import SelfPlay
from NNet import NNet

from Config import TRAINING_ITERATIONS, \
    SAVE_THRESHOLD


class Train:
    def __init__(self, iterations=TRAINING_ITERATIONS, save_threshold=SAVE_THRESHOLD):
        self.iterations = iterations
        self.save_threshold = save_threshold
        self.nn = NNet()
        self.sp = SelfPlay()
        self.arena = Arena()

        self.nn.create_model('best')

    def _iteration(self):
        history = self.sp.generate_data()
        self.nn.train(history, './model/best.h5')
        wins, draw, loses = self.arena.play_games()
        if wins / (wins + loses) > self.save_threshold:
            copy('./model/latest.h5', './model/best.h5')
            print("Saved new model")

    def start(self):
        for _ in tqdm(range(self.iterations), desc="Training"):
            self._iteration()


if __name__ == '__main__':
    Train().start()
    # Arena().play_games()
    # copy('./model/latest.h5', './model/best.h5')
