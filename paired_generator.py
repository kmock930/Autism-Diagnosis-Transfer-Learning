import tensorflow as tf
import numpy as np

from pair import final_tf


class PairedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, generator1, generator2):
        self.generator1 = generator1
        self.generator2 = generator2
        self.length = min(len(generator1), len(generator2))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        X1, y1 = self.generator1[index]
        X2, y2 = self.generator2[index]
        # Use the `final_tf` function to create pairs
        X1_paired, X2_paired = final_tf(X1, X2, y1, y2)
        # For labels, we can concatenate y1 and y2
        y_paired = np.concatenate([y1, y2], axis=0)
        return (X1_paired, X2_paired), y_paired
