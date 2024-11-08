import tensorflow as tf
import numpy as np


def balanced_batch_generator(dataset, labels, n_classes, n_samples):
    label_set = np.unique(labels)
    label_to_indices = {label: np.where(labels == label)[0] for label in label_set}
    used_label_indices_count = {label: 0 for label in label_set}

    while True:
        classes = np.random.choice(label_set, n_classes, replace=False)
        batch_indices = []
        for class_ in classes:
            indices = label_to_indices[class_]
            start = used_label_indices_count[class_]
            end = start + n_samples
            if end > len(indices):
                np.random.shuffle(indices)
                used_label_indices_count[class_] = 0
                start = 0
                end = n_samples
            batch_indices.extend(indices[start:end])
            used_label_indices_count[class_] += n_samples
        batch_data = dataset[batch_indices]
        batch_labels = labels[batch_indices]
        yield batch_data, batch_labels
