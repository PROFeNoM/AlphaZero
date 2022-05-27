from keras.callbacks import LearningRateScheduler
from keras.models import load_model
from keras import backend as K
import numpy as np
import pickle
from NNet import NNet, opt, history_logger

TRAINING_EPOCHS = 50


def load_samples():
    print("Loading samples...")
    with open('./data/samples.history', mode='rb') as f:
        return pickle.load(f)


def train_network():
    history = load_samples()
    xs, y_policies, y_values = zip(*history)

    xs = np.array(xs)
    y_policies = np.array(y_policies)
    y_values = np.array(y_values)

    print("Loading network...")
    model = load_model('./model/trained.h5')
    print("Network loaded.")
    model.compile(loss={'v': 'mean_squared_error', 'pi': 'categorical_crossentropy'},
                  optimizer=opt,
                  metrics=['accuracy'])
    print("Network compiled.")

    def step_decay(epoch):
        x = 0.05
        if epoch >= 20: x = 1e-2
        if epoch >= 40: x = 1e-3
        return x

    lr_decay = LearningRateScheduler(step_decay)
    print("Learning Rate Scheduler initialized.")
    epoch_count = TRAINING_EPOCHS

    model.fit(xs, [y_policies, y_values],
              batch_size=2048,
              epochs=epoch_count,
              verbose=1,
              callbacks=[lr_decay, history_logger])

    model.save('./model/trained.h5')

    K.clear_session()
    del model


if __name__ == '__main__':
    print("Creating network...")
    NNet().create_model('trained')
    print("Network created.")
    train_network()
    print("Supervised training done.")
