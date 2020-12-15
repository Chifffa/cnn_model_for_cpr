from typing import Tuple

import keras
import numpy as np
import matplotlib.pyplot as plt
import cv_utils.augmentation as aug

from config import TRAIN_DATA_PATH, TEST_DATA_PATH, BATCH_SIZE, CLASS_NAMES, NUM_CLASSES, INPUT_SHAPE


class DataGenerator(keras.utils.Sequence):
    def __init__(self, is_train: bool = True):
        """
        Data generator for the task of classifying a correlation signal as true or false.

        :param is_train: if True, generating train data with performing augmentation and every epoch shuffling.
            Else generating test data without performing augmentation and every epoch shuffling.
        """
        self.is_train = is_train

        self.batch_size = BATCH_SIZE
        self.input_shape = INPUT_SHAPE
        self.classes = CLASS_NAMES
        self.num_classes = NUM_CLASSES

        # Loading prepared data. Format is [(img, label_onehot), (..., ...), ...].
        if self.is_train:
            self.data = np.load(TRAIN_DATA_PATH, allow_pickle=True).item()[0]
        else:
            self.data = np.load(TEST_DATA_PATH, allow_pickle=True).item()[0]

        augmentations = [
            aug.FlipLR(0.5),
            aug.FlipUD(0.5),
            aug.OneOf([aug.AddUniformNoise(0.5, -0.05, 0.05), aug.MultUniformNoise(0.5)]),
            aug.Rotation(0.5, min_angle=-15, max_angle=15)
        ]
        self.aug = aug.Sequential(augmentations)
        self.on_epoch_end(init=True)

    def on_epoch_end(self, init: bool = False) -> None:
        """
        Random shuffling of training data at the end of each epoch.

        :param init: use init = True to shuffle train and test data after DataGenerator object initialization.
        """
        if init or self.is_train:
            np.random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data) // self.batch_size + (len(self.data) % self.batch_size != 0)

    def __getitem__(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Making batch.

        :param batch_idx: batch number.
        :return: image tensor and labels tensor.
        """
        batch = self.data[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        images = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        labels = np.zeros((self.batch_size, self.num_classes))
        for i in range(self.batch_size):
            img = batch[i][0].copy()
            if self.is_train:
                img = self.aug.augment(img)
            images[i, :, :, :] = img
            labels[i, :] = batch[i][1].copy()
        return np.float32(images), np.float32(labels)

    def show(self, batch_idx: int) -> None:
        """
        Method for showing original and augmented image with labels.

        :param batch_idx: batch number.
        """
        batch = self.data[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        for i in range(self.batch_size):
            img_original = batch[i][0].copy()
            img_augmented = self.aug.augment(img_original.copy())
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
            plt.subplot(axes[0])
            plt.imshow(img_original[:, :, 0])
            plt.title('Original, class = "{}"'.format(self.classes[int(batch[i][1][1])]))
            plt.subplot(axes[1])
            plt.imshow(img_augmented[:, :, 0])
            plt.title('Augmented, class = "{}"'.format(self.classes[int(batch[i][1][1])]))
            if plt.waitforbuttonpress(0):
                plt.close('all')
                raise SystemExit
            plt.close(fig)


def test_data_generator(is_train: bool = True) -> None:
    """
    Function for testing data generator. Visualizing original and augmented images with labels.
    Mouse click to continue, press any button to exit.

    :param is_train: if True, generating train data. Else generating test data.
    """
    data_gen = DataGenerator(is_train)
    for index, _ in enumerate(data_gen):
        data_gen.show(index)
