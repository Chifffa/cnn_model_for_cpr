import os
import argparse

import numpy as np
from tqdm import tqdm

from config import TRAIN_DATA_PATH, TEST_DATA_PATH, INPUT_SHAPE


def prepare_dataset(corr_path: str, with_holo: bool = False, fsf: int = 4):
    """
    Creating a dataset to classify correlation signals from correlation matrices obtained in various simulations.

    :param corr_path: path to folder with obtained correlation matrices (matrices are files with "corr" in name and
        extension ".npy").
    :param with_holo: set to True if signals were obtained using holographic correlation filters.
    :param fsf: field size factor of images when using holograms (usually fsf = 4).
    """
    data_names = [x for x in os.listdir(corr_path) if 'corr' in x and x.endswith('.npy')][::-1]
    train, train_labels, test, test_labels = [], [], [], []
    for data_name in tqdm(data_names, desc='Files'):
        data = np.load(os.path.join(corr_path, data_name), allow_pickle=True)
        if with_holo:
            height, width = data.shape[1:]
            h = int(height * (3 / 4 - 1 / fsf / 2))
            w = int(width * (3 / 4 - 1 / fsf / 2))
            data = data[:, h:h + height // fsf, w:w + width // fsf]
        data = np.abs(data)
        h, w = data.shape[1:]

        inp_h_shift = INPUT_SHAPE[0] // 2
        inp_w_shift = INPUT_SHAPE[1] // 2

        class_name = data_name.split('___')[0].split('corr_')[-1]

        image_data = data[:, h // 2 - inp_h_shift:h // 2 + inp_h_shift, w // 2 - inp_w_shift:w // 2 + inp_w_shift]
        image_data = np.expand_dims(image_data, axis=-1)
        if class_name == 'train':
            train_labels.append(np.stack([np.ones(data.shape[0]), np.zeros(data.shape[0])], axis=1))
            train.append(image_data)
        elif class_name in ['false_1', 'false_2']:
            train_labels.append(np.stack([np.zeros(data.shape[0]), np.ones(data.shape[0])], axis=1))
            train.append(image_data)
        elif class_name == 'test':
            test_labels.append(np.stack([np.ones(data.shape[0]), np.zeros(data.shape[0])], axis=1))
            test.append(image_data)
        else:
            test_labels.append(np.stack([np.zeros(data.shape[0]), np.ones(data.shape[0])], axis=1))
            test.append(image_data)

    train = np.abs(np.concatenate(train, axis=0))
    train = train / np.max(train, axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]
    train_labels = np.concatenate(train_labels, axis=0)
    test = np.abs(np.concatenate(test, axis=0))
    test = test / np.max(test, axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]
    test_labels = np.concatenate(test_labels, axis=0)

    train_out = []
    for i in range(train.shape[0]):
        train_out.append((train[i, :, :, :], train_labels[i, :]))
    test_out = []
    for i in range(test.shape[0]):
        test_out.append((test[i, :, :, :], test_labels[i, :]))
    np.save(TRAIN_DATA_PATH, {0: train_out})
    np.save(TEST_DATA_PATH, {0: test_out})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corr_path', type=str, help='Path to folder with correlation matrices.')
    parser.add_argument('--holo', action='store_true', help='Use hologram processing for correlation outputs.')
    parser.add_argument('--fsf', type=int, default=4, help='Field size factor for holograms.')
    args = parser.parse_args()

    prepare_dataset(args.corr_path, args.holo, args.fsf)
