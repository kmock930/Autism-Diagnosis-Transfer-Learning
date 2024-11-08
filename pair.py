import tensorflow as tf
import numpy as np

def arrange_tf(v1, v2, l1, l2):
    n = tf.shape(v1)[0]
    m = n // 2

    # Split v1 into two halves
    video1_1 = v1[:m]
    video1_2 = v1[m:]

    # Convert labels to lists if they are tensors
    if isinstance(l1, tf.Tensor):
        l1 = l1.numpy()
    if isinstance(l2, tf.Tensor):
        l2 = l2.numpy()

    # Check if labels are different
    if l1.tolist() != l2.tolist():
        # Split v2 into two halves
        video2_1 = v2[:m]
        video2_2 = v2[m:]
    else:
        # Swap halves of v2
        video2_1 = v2[m:]
        video2_2 = v2[:m]

    return video1_1, video1_2, video2_1, video2_2

def make_pair_tf(p1, p2):
    len_p1 = tf.shape(p1)[0]  # Batch size of p1
    len_p2 = tf.shape(p2)[0]  # Batch size of p2

    # Expand dimensions to facilitate broadcasting
    p1_expanded = tf.expand_dims(p1, 1)  # Shape: (len_p1, 1, ...)
    p2_expanded = tf.expand_dims(p2, 0)  # Shape: (1, len_p2, ...)

    # Compute the rank (number of dimensions) of the expanded tensors
    rank = tf.rank(p1_expanded)

    # Create the multiples vector for tiling
    multiples_p1 = tf.concat([[1, len_p2], tf.ones(rank - 2, dtype=tf.int32)], axis=0)
    multiples_p2 = tf.concat([[len_p1, 1], tf.ones(rank - 2, dtype=tf.int32)], axis=0)

    # Tile the tensors
    p1_tiled = tf.tile(p1_expanded, multiples_p1)
    p2_tiled = tf.tile(p2_expanded, multiples_p2)

    # Reshape to get pairs
    p1_shape = tf.shape(p1)
    new_shape = tf.concat([[-1], p1_shape[1:]], axis=0)
    new_vid1 = tf.reshape(p1_tiled, new_shape)
    new_vid2 = tf.reshape(p2_tiled, new_shape)

    return new_vid1, new_vid2

def final_tf(v1, v2, l1, l2):
    v1_1, v1_2, v2_1, v2_2 = arrange_tf(v1, v2, l1, l2)
    modal1, modal2 = make_pair_tf(v1_1, v2_1)
    mod1, mod2 = make_pair_tf(v1_2, v2_2)
    video1 = tf.concat([modal1, mod1], axis=0)
    video2 = tf.concat([modal2, mod2], axis=0)
    return video1, video2
