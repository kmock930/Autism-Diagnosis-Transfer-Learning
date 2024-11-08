import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import layers, models, optimizers, losses, metrics


def load_i3d_model(input_shape=(64, 224, 224, 3)):
    # 加载预训练的 I3D 模型
    i3d_model = hub.KerasLayer("https://tfhub.dev/deepmind/i3d-kinetics-400/1", trainable=False)

    # 定义输入
    inputs = layers.Input(shape=input_shape)

    # I3D 模型返回的 logits 有 400 个类别（Kinetics-400）
    logits = i3d_model(inputs)  # 输入是一个字典

    features = layers.Dense(512, activation='relu')(logits)
    # 构建模型
    model = models.Model(inputs=inputs, outputs=features)

    return model


def supervised_contrastive_loss(labels, features, temperature=0.07):
    # 特征归一化
    features = tf.math.l2_normalize(features, axis=1)
    batch_size = tf.shape(features)[0]

    # 计算相似度矩阵
    similarity_matrix = tf.matmul(features, features, transpose_b=True)  # (batch_size, batch_size)

    # 获取标签相等的矩阵
    labels = tf.reshape(labels, (batch_size, 1))
    labels_equal = tf.equal(labels, tf.transpose(labels))  # (batch_size, batch_size)
    labels_equal = tf.cast(labels_equal, tf.float32)

    # 掩码，排除自身
    mask = tf.cast(tf.eye(batch_size), tf.bool)
    labels_equal = tf.where(mask, tf.zeros_like(labels_equal), labels_equal)
    similarity_matrix = tf.where(mask, tf.zeros_like(similarity_matrix), similarity_matrix)

    # 计算分子和分母
    exp_similarity = tf.exp(similarity_matrix / temperature)
    sum_exp = tf.reduce_sum(exp_similarity, axis=1, keepdims=True)
    pos_exp = tf.reduce_sum(exp_similarity * labels_equal, axis=1)

    loss = -tf.math.log(pos_exp / (sum_exp - exp_similarity.diagonal()))
    loss = tf.reduce_mean(loss)
    return loss



def train_model(model, train_generator, val_generator, epochs=10, temperature=0.07):
    optimizer = optimizers.Adam(learning_rate=1e-4)
    train_loss_results = []
    val_loss_results = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_val_loss_avg = tf.keras.metrics.Mean()

        # 训练循环
        for step, (X_batch, y_batch) in enumerate(train_generator):
            with tf.GradientTape() as tape:
                features = model(X_batch, training=True)
                loss = supervised_contrastive_loss(y_batch, features, temperature)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss_avg.update_state(loss)

        # 记录训练损失
        train_loss = epoch_loss_avg.result()
        train_loss_results.append(train_loss)

        # 验证循环
        for X_batch_val, y_batch_val in val_generator:
            features_val = model(X_batch_val, training=False)
            val_loss = supervised_contrastive_loss(y_batch_val, features_val, temperature)
            epoch_val_loss_avg.update_state(val_loss)

        # 记录验证损失
        val_loss = epoch_val_loss_avg.result()
        val_loss_results.append(val_loss)

        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    return train_loss_results, val_loss_results


def evaluate_model(model, train_generator, val_generator, num_classes):
    # 冻结模型
    for layer in model.layers:
        layer.trainable = False

    # 添加线性分类器
    inputs = model.input
    features = model.output
    outputs = layers.Dense(num_classes, activation='softmax')(features)
    classifier = models.Model(inputs=inputs, outputs=outputs)

    # 编译模型
    classifier.compile(
        loss=losses.CategoricalCrossentropy(),
        optimizer=optimizers.Adam(learning_rate=1e-4),
        metrics=[metrics.CategoricalAccuracy()]
    )

    # 训练分类器
    classifier.fit(
        train_generator,
        validation_data=val_generator,
        epochs=5
    )

    # 评估模型
    results = classifier.evaluate(val_generator)
    print(f"Validation Loss: {results[0]}, Validation Accuracy: {results[1]}")