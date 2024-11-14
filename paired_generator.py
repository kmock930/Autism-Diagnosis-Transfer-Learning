import tensorflow as tf
import numpy as np

from pair import final_tf


class PairedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, generator1, generator2, batch_size=16):
        self.generator1 = generator1
        self.generator2 = generator2
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return min(len(self.generator1), len(self.generator2))

    def __getitem__(self, index):
        X1, y1 = self.generator1[index]
        X2, y2 = self.generator2[index]
        # 根据标签匹配正样本对，非匹配的为负样本对
        positive_pairs = []
        negative_pairs = []
        for i in range(len(y1)):
            for j in range(len(y2)):
                if y1[i] == y2[j]:
                    positive_pairs.append((X1[i], X2[j], y1[i]))
                else:
                    negative_pairs.append((X1[i], X2[j], y1[i]))
        # 组合正负样本对
        X1_batch = np.array([pair[0] for pair in positive_pairs + negative_pairs])
        X2_batch = np.array([pair[1] for pair in positive_pairs + negative_pairs])
        y_batch = np.array([pair[2] for pair in positive_pairs + negative_pairs])
        return (X1_batch, X2_batch), y_batch

    def on_epoch_end(self):
        pass
