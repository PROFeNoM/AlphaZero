import pickle

import numpy as np
from keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from keras.models import Model
from keras.optimizer_v2.gradient_descent import SGD
from keras.regularizers import l2
from keras.backend import clear_session
import os

from keras.callbacks import CSVLogger
from keras.models import load_model

import go

NETWORK_FILTERS = 64  # original = 256
NETWORK_RESIDUAL_NUM = 9  # original = 19
NETWORK_INPUT_SHAPE = (go.N, go.N, 9)  # original = (go.N, go.N, 17)
NETWORK_OUTPUT_SIZE = (go.N * go.N) + 1

TRAINING_EPOCHS = 15
TRAINING_LOG_PATH = 'training_log.csv'

opt = SGD(learning_rate=0.001, momentum=0.9)
history_logger = CSVLogger(TRAINING_LOG_PATH, separator=',', append=True)


class NNet:
    def __init__(self, network_filters=NETWORK_FILTERS, network_residual_num=NETWORK_RESIDUAL_NUM,
                 network_input_shape=NETWORK_INPUT_SHAPE, network_output_size=NETWORK_OUTPUT_SIZE,
                 training_epochs=TRAINING_EPOCHS):
        self.network_filters = network_filters
        self.network_residual_num = network_residual_num
        self.network_input_shape = network_input_shape
        self.network_output_size = network_output_size
        self.training_epochs = training_epochs

    def create_model(self, name):
        if os.path.exists(f"./model/{name}.h5"):
            return

        input = Input(shape=self.network_input_shape)

        x = self.conv(self.network_filters)(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        for i in range(self.network_residual_num):
            x = self.residual_block()(x)

        p = self.conv_size_one(2)(x)
        p = BatchNormalization()(p)
        p = Activation('relu')(p)

        p = GlobalAveragePooling2D()(p)

        p = Dense(self.network_output_size, kernel_regularizer=l2(1e-4),
                  activation='softmax', name='pi')(p)

        v = self.conv_size_one(1)(x)
        v = BatchNormalization()(v)
        v = Activation('relu')(v)

        v = GlobalAveragePooling2D()(v)

        v = Dense(1, kernel_regularizer=l2(1e-4))(v)
        v = Activation('tanh', name='v')(v)

        model = Model(inputs=input, outputs=[p, v])

        os.makedirs('./model', exist_ok=True)
        model.save(f"./model/{name}.h5")

        clear_session()
        del model

    def conv(self, filters):
        return Conv2D(filters, 3, padding='same',
                      kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))

    def conv_size_one(self, filters):
        return Conv2D(filters, 1, padding='same',
                      kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))

    def residual_block(self):
        def f(x):
            sc = x
            x = self.conv(self.network_filters)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = self.conv(self.network_filters)(x)
            x = BatchNormalization()(x)
            x = Add()([x, sc])
            x = Activation('relu')(x)
            return x

        return f

    def train(self, data, model_path):
        x, y_p, y_v = zip(*data)

        x = np.array(x)
        y_p = np.array(y_p)
        y_v = np.array(y_v)

        model = load_model(model_path)
        model.compile(loss={'v': 'mean_squared_error', 'pi': 'categorical_crossentropy'},
                      optimizer=opt,
                      metrics=['accuracy'], )

        model.fit(x, [y_p, y_v],
                  batch_size=32,
                  epochs=self.training_epochs,
                  verbose=1,
                  callbacks=[history_logger])

        model.save('./model/latest.h5',
                   overwrite=True,
                   include_optimizer=False)

        clear_session()
        del model


if __name__ == '__main__':
    pass
    #with open('./data/1653430470.213831.npy', mode='rb') as f:
    #    history = pickle.load(f)
    #    NNet().train(history, './model/best.h5')
