import logging
import warnings
import os
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from games.tablut.players.alpha_zero import Model


def main():
    model = Model()
    model.train_model(epochs=1, step_for_epoch=1700, batch_size=128)
    model.save_model()


if __name__ == '__main__':
    main()


