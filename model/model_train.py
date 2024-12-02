"""Adapted from MDSupCL/model/model_train.py (https://github.com/asharani97/MDSupCL/tree/main/model)"""
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks,metrics
from tqdm import tqdm


def load_c3d_model(input_shape=(12, 64, 64, 3), feature_dim=512):
    """Adapted from https://github.com/hx173149/C3D-tensorflow/blob/master/c3d_model.py"""
    # Define inputs
    inputs = layers.Input(shape=input_shape)

    # Layer 1 convolution and pooling
    x = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)

    # Layer 2 convolution and pooling
    x = layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

    # Layer 3 and 4 convolution and pooling
    x = layers.Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)

    # Fifth and Sixth Layer Convolution and Pooling
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

    # Seventh and Eighth Layer Convolution and Pooling
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Output feature vector
    features = layers.Dense(feature_dim, activation='relu')(x)

    model = models.Model(inputs=inputs, outputs=features)

    return model


def supervised_contrastive_loss(labels, features, dataset_ids, temperature=0.07):
    # Normalize features
    features = tf.math.l2_normalize(features, axis=1)
    batch_size = tf.shape(features)[0]

    # Compute similarity matrix
    similarity_matrix = tf.matmul(features, features, transpose_b=True)  # (batch_size, batch_size)

    # Reshape labels and dataset_ids
    labels = tf.reshape(labels, (batch_size, 1))
    dataset_ids = tf.reshape(dataset_ids, (batch_size, 1))

    # Create masks
    labels_equal = tf.equal(labels, tf.transpose(labels))  # Same labels
    datasets_different = tf.not_equal(dataset_ids, tf.transpose(dataset_ids))  # Different datasets

    # Positive mask: same label, different dataset
    positive_mask = tf.logical_and(labels_equal, datasets_different)

    # Negative mask: different label, different dataset
    negative_mask = tf.logical_and(tf.logical_not(labels_equal), datasets_different)

    # Exclude self-comparisons
    mask = tf.logical_not(tf.eye(batch_size, dtype=tf.bool))
    positive_mask = tf.logical_and(positive_mask, mask)
    negative_mask = tf.logical_and(negative_mask, mask)

    # Numerator and Denominator
    exp_similarity = tf.exp(similarity_matrix / temperature)
    numerator = tf.reduce_sum(exp_similarity * tf.cast(positive_mask, tf.float32), axis=1)
    denominator = numerator + tf.reduce_sum(exp_similarity * tf.cast(negative_mask, tf.float32), axis=1)

    # Avoid division by zero
    epsilon = 1e-12
    loss = -tf.math.log((numerator + epsilon) / (denominator + epsilon))
    loss = tf.reduce_mean(loss)
    return loss


def train_msupcl_model(model, train_generator, epochs=10, temperature=0.07):
    # using adam as optimizers
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss_history = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        # Calculation of the weighted average of the loss
        epoch_loss_avg = tf.keras.metrics.Mean()
        for step in tqdm(range(len(train_generator)), desc=f"Epoch {epoch + 1}/{epochs}"):
            # load data
            X_batch, y_batch, dataset_ids = train_generator[step]
            with tf.GradientTape() as tape:
                # extract features
                features = model(X_batch, training=True)
                # calculate loss
                loss = supervised_contrastive_loss(y_batch, features, dataset_ids, temperature)
            # calculate gradient by using GradientTape
            gradients = tape.gradient(loss, model.trainable_variables)
            # update trainable vars
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            # update loss
            epoch_loss_avg.update_state(loss)
        epoch_loss = epoch_loss_avg.result().numpy()
        print(f"Training Loss: {epoch_loss:.4f}")
        loss_history.append(epoch_loss)
    return loss_history



def linear_evaluation(model, train_generator, test_generator1, test_generator2, num_classes=2, num_epochs=5):
    # learning rate reduced in each 3 epoch
    lr_scheduler = callbacks.LearningRateScheduler(scheduler)
    # freeze feature extractor
    for layer in model.layers:
        layer.trainable = False
    features = model.output
    # add classification layer
    outputs = layers.Dense(num_classes, activation='softmax')(features)
    classifier_model = models.Model(inputs=model.input, outputs=outputs)
    # compile model
    classifier_model.compile(
        loss=losses.SparseCategoricalCrossentropy(),
        optimizer=optimizers.Adam(learning_rate=1e-4),
        metrics=[metrics.SparseCategoricalAccuracy()]
    )
    # train
    history = classifier_model.fit(
        train_generator,
        epochs=num_epochs,
        callbacks=[lr_scheduler],
    )
    print("Evaluating on Violence Test Set:")
    results_violence = classifier_model.evaluate(test_generator1)
    print(f"Violence Test Loss: {results_violence[0]}, Test Accuracy: {results_violence[1]}")
    print("Evaluating on TikTok Test Set:")
    results_tiktok = classifier_model.evaluate(test_generator2)
    print(f"TikTok Test Loss: {results_tiktok[0]}, Test Accuracy: {results_tiktok[1]}")
    return results_violence, results_tiktok, history, classifier_model

def scheduler(epoch, lr):
    # Every 3 epochs, the learning rate decays to 0.01 times the original
    if epoch % 3 == 0 and epoch > 0:
        return lr * 0.01
    return lr


def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = tf.shape(z_i)[0]

    # Normalized feature vector
    z_i = tf.math.l2_normalize(z_i, axis=1)
    z_j = tf.math.l2_normalize(z_j, axis=1)

    # Combined Features Vector
    z = tf.concat([z_i, z_j], axis=0)

    # Compute the similarity matrix
    similarity_matrix = tf.matmul(z, z, transpose_b=True)
    similarity_matrix = similarity_matrix / temperature

    # Create masks to exclude self-comparison
    mask = tf.eye(2 * batch_size)
    logits = similarity_matrix - mask * 1e9

    # Create labels
    positive_indices = tf.concat([tf.range(batch_size, 2 * batch_size), tf.range(0, batch_size)], axis=0)
    labels = positive_indices

    # Calculating cross-entropy loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(loss)
    return loss

def load_c3d_sscl_model(input_shape=(12, 64, 64, 3), feature_dim=512):
    # Defining Inputs
    inputs = layers.Input(shape=input_shape)

    # Layer 1 convolution and pooling
    x = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)

    # Layer 2 convolution and pooling
    x = layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

    # Layer 3 and 4 convolution and pooling
    x = layers.Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)

    # Layer 5 and 6 Convolution and Pooling
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

    # Layer 7 and 8 Convolution and Pooling
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output feature vector
    features = layers.Dense(feature_dim, activation='relu', name='features')(x)

    # Output feature head projections
    projections = layers.Dense(feature_dim, name='projections')(features)
    model = models.Model(inputs=inputs, outputs=[features, projections])

    return model

def train_simclr_model(model, train_generator, epochs=10, temperature=0.5):
    # learning rate reduced in each 3 epoch
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss_history = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        # Calculation of the weighted average of the loss
        epoch_loss_avg = tf.keras.metrics.Mean()
        for step in tqdm(range(len(train_generator)), desc=f"Epoch {epoch + 1}/{epochs}"):
            # load data
            (x_i, x_j), _ = train_generator[step]
            with tf.GradientTape() as tape:
                # extract features projections
                _, z_i = model(x_i, training=True)
                _, z_j = model(x_j, training=True)
                # calculate loss
                loss = nt_xent_loss(z_i, z_j, temperature)
            # calculate gradient by using GradientTape
            gradients = tape.gradient(loss, model.trainable_variables)
            # update trainable vars
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            # update loss
            epoch_loss_avg.update_state(loss)
        epoch_loss = epoch_loss_avg.result().numpy()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
        loss_history.append(epoch_loss)
    return loss_history

def linear_evaluation_sscl(model, train_generator, val_generator, test_generator, num_classes=2, num_epochs=3):
    # learning rate reduced in each 3 epoch
    lr_scheduler = callbacks.LearningRateScheduler(scheduler)
    # freeze feature extractor
    for layer in model.layers:
        layer.trainable = False
    inputs = model.input
    # read feature from network
    features = model.outputs[0]
    # add classification layer
    outputs = layers.Dense(num_classes, activation='softmax')(features)
    classifier_model = models.Model(inputs=inputs, outputs=outputs)
    classifier_model.compile(
        loss=losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=[metrics.SparseCategoricalAccuracy()]
    )
    # train model
    history = classifier_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=num_epochs,
        callbacks=[lr_scheduler],
    )
    # evaluate
    results = classifier_model.evaluate(test_generator)
    print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")
    return results, history, classifier_model