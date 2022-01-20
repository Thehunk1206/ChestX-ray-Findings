'''
MIT License

Copyright (c) 2022 Tauhid Khan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from glob import glob
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.python.data.ops.dataset_ops import DatasetV2

import pandas as pd
import numpy as np

class TfdataPipeline:
    def __init__(
        self,
        BASE_DATASET_DIR: str,
        IMG_H: int = 224,
        IMG_W: int = 224,
        IMG_C: int = 3,
        batch_size: int = 8,
    ) -> None:
        if not os.path.exists(BASE_DATASET_DIR):
            raise ValueError('The dataset directory does not exist')
        
        self.BASE_DATASET_DIR = BASE_DATASET_DIR
        self.IMG_H = IMG_H
        self.IMG_W = IMG_W
        self.IMG_C = IMG_C
        self.batch_size = batch_size

    def _load_image_path_and_labels(self, path_to_csv:str) -> Tuple[tf.Tensor, tf.Tensor]:
        '''
        The CSV file should be in the following format:
        image_name,label1,label2,label3,label4...

        where image_name is the name of the image and label1, label2, label3, label4... are the labels either 1 or 0
        '''
        if not os.path.exists(path_to_csv):
            raise ValueError('The csv file does not exist')
        
        dataset_df = pd.read_csv(path_to_csv)

        labels_ = dataset_df.iloc[:, 1:].values # Get all the labels as a numpy array
        labels_ = tf.convert_to_tensor(labels_, dtype=tf.float32)

        image_paths_ = dataset_df.iloc[:, 0].values # Get all the image paths as a numpy array
        image_paths_ = tf.convert_to_tensor(image_paths_, dtype=tf.string)
        base_dir = tf.constant(f'{self.BASE_DATASET_DIR}images/')
        image_paths_ = tf.strings.join([base_dir, image_paths_])
        return image_paths_, labels_
    
    def _read_image(self, image_path: tf.Tensor) -> tf.Tensor:
        '''
        This function is used to read the image from the path
        args:
            image_path: tf.Tensor, The path to the image
        returns:
            tf.Tensor, The image
        '''
        image_string    = tf.io.read_file(image_path)
        image           = tf.io.decode_image(image_string, channels=1, expand_animations=False)
        image           = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image           = tf.image.resize(image, [self.IMG_H, self.IMG_W], method = tf.image.ResizeMethod.BICUBIC)
        image           = tf.image.grayscale_to_rgb(image)
        return image

    def _augment(self, image:tf.Tensor) -> Sequential:
        '''
        This function is used to augment the image by randomly rotating and randomly changing Contrast 
        args:
            image:tf.Tensor, The image to be augmented
        returns:
            tf.Tensor, The augmented image
        '''
        aug = Sequential([
            layers.RandomRotation(factor = 0.05, seed = 42, fill_mode = 'constant', fill_value = 0.0),
            layers.RandomContrast(factor = 0.2, seed = 42),
        ])
        return aug(image, training=True)
    
    def _tf_dataset(self, image_path: tf.Tensor, labels: tf.Tensor) -> DatasetV2:
        '''
        Creates a tf.data.Dataset object from the image path and labels
        args:
            image_path: tf.Tensor, The image path
            labels: tf.Tensor, The labels
        returns:
            tf.data.Dataset, The tf.data.Dataset object which will be consumed by the model
        '''
        dataset = tf.data.Dataset.from_tensor_slices((image_path, labels))
        dataset = (dataset
                        .map(lambda x, y: (self._augment(self._read_image(x)), y), 
                        num_parallel_calls=tf.data.AUTOTUNE)
                        .shuffle(buffer_size=256)
                        .batch(self.batch_size)
                        .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        return dataset

    def load_dataset(self, path_to_csv: str) -> DatasetV2:
        '''
        This function is used to get the dataset
        args:
            path_to_csv: str, The path to the csv file
        returns:
            tf.data.Dataset, The dataset
        '''
        path_to_csv = os.path.join(self.BASE_DATASET_DIR, path_to_csv)
        image_paths, labels = self._load_image_path_and_labels(path_to_csv)
        dataset = self._tf_dataset(image_paths, labels)
        return dataset

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    tf_dataset = TfdataPipeline(BASE_DATASET_DIR='./sample_dataset/', IMG_H=224, IMG_W=224, IMG_C=3, batch_size=1)
    dataset = tf_dataset.load_dataset('train_labels.csv')
    for image, label in dataset.take(1):
        print(image.shape, label.shape)