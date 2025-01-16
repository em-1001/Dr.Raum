# dataset.py

import config
import numpy as np
import keras
import os
import cv2
import nibabel as nib
import tensorflow as tf


class UnetrDataset(keras.utils.Sequence):
    def __init__(self, list_IDs, transform=None, dim=(config.IMG_SIZE,config.IMG_SIZE), n_channels = 2, shuffle=True):
        self.dim = dim
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        id = self.list_IDs[index]

        X = np.zeros((config.VOLUME_SLICES, *self.dim, self.n_channels))
        y = np.zeros((config.VOLUME_SLICES, 240, 240))
        Y = np.zeros((config.VOLUME_SLICES, *self.dim, 4))

        case_path = os.path.join(config.TRAIN_DATASET_PATH, id)

        data_path = os.path.join(case_path, f'{id}_flair.nii');
        flair = nib.load(data_path).get_fdata()

        data_path = os.path.join(case_path, f'{id}_t1ce.nii');
        t1ce = nib.load(data_path).get_fdata()

        data_path = os.path.join(case_path, f'{id}_seg.nii');
        seg = nib.load(data_path).get_fdata()

        for j in range(config.VOLUME_SLICES):
            X[j,:,:,0] = cv2.resize(flair[:,:,j+config.VOLUME_START_AT], (config.IMG_SIZE, config.IMG_SIZE));
            X[j,:,:,1] = cv2.resize(t1ce[:,:,j+config.VOLUME_START_AT], (config.IMG_SIZE, config.IMG_SIZE));

            y[j] = seg[:,:,j+config.VOLUME_START_AT];

        # Generate masks
        y[y==4] = 3;
        # mask: (VOLUME_SLICES, 240, 240, 4)
        mask = tf.one_hot(y, 4);
        Y = tf.image.resize(mask, (config.IMG_SIZE, config.IMG_SIZE));


        # X: (VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2)
        # Y: (VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 4)
        x, y = X/np.max(X), Y.numpy()

        if self.transform:
            rand_affine = self.transform
            rand_affine.set_random_state()
            x = np.transpose(x, (3, 0, 1, 2))  # (96, 128, 128, 2) -> (2, 96, 128, 128)
            y = np.transpose(y, (3, 0, 1, 2))  # (96, 128, 128, 4) -> (4, 96, 128, 128)
            data_dict = {"image": x, "label": y}
            affined_data_dict = rand_affine(data_dict)
            x, y = affined_data_dict["image"], affined_data_dict["label"]
            x = np.transpose(x, (1, 2, 3, 0))  # (2, 96, 128, 128) -> (96, 128, 128, 2)
            y = np.transpose(y, (1, 2, 3, 0))  # (4, 96, 128, 128) -> (96, 128, 128, 4)

        return x, y
