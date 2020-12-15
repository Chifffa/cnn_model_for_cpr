import os
import argparse
from multiprocessing import cpu_count
from typing import Optional

import keras
from cv_utils import get_date, LogCallback, session_config, init_random_generators
from cv_utils.classification import Recall, Precision, F1Score

from cpr_cnn_model import get_model, DataGenerator
from config import LEARNING_RATE, NUM_CLASSES, EPOCHS, LOGS_DIR


def train(save_path: str, weights: Optional[str]) -> None:
    """
    Training classifier of correlation signals.

    :param save_path: path to save weights and training logs.
    :param weights: path to saved model weights.
    """
    log_dir = os.path.join(save_path, LOGS_DIR)
    os.makedirs(log_dir, exist_ok=True)

    train_data_gen = DataGenerator(is_train=True)
    test_data_gen = DataGenerator(is_train=False)
    model = get_model(weights)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  metrics=[Recall(NUM_CLASSES), Precision(NUM_CLASSES), F1Score(NUM_CLASSES), 'accuracy'])
    model.summary()

    with LogCallback(save_path, log_dir) as log_callback:
        callbacks = [keras.callbacks.TensorBoard(log_dir=log_dir), log_callback]
        model.fit_generator(generator=train_data_gen, validation_data=test_data_gen, validation_freq=1,
                            validation_steps=len(test_data_gen), epochs=EPOCHS, callbacks=callbacks,
                            workers=min(24, cpu_count()))


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser('Script for training classifier for correlation signals.')
    parser.add_argument('-sp', '--save_path', type=str, default='results_{}'.format(get_date()),
                        help='Directory to save weights and training logs.')
    parser.add_argument('-d', '--device', type=int, default=0, help='GPU device index.')
    parser.add_argument('--weights', type=str, default=None, help='Path to saved model.')
    parser.add_argument('--cpu', action='store_true', help='Use only cpu.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random generators seed. If seed < 0, seed will be None.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.seed >= 0:
        init_random_generators(args.seed)
    if args.cpu:
        session_config(None, set_keras=True)
    else:
        session_config([args.device], set_keras=True)

    train(args.save_path, args.weights)
