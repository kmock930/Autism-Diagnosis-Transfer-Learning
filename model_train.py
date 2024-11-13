import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import layers, models, optimizers, losses, metrics


def load_c3d_model(input_shape=(16, 112, 112, 3), feature_dim=512):
    # 定义输入
    inputs = layers.Input(shape=input_shape)

    # 第一层卷积和池化
    x = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)

    # 第二层卷积和池化
    x = layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

    # 第三、四层卷积和池化
    x = layers.Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

    # 第五、六层卷积和池化
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

    # 第七、八层卷积和池化
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

    # 展平和全连接层
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # 输出特征向量
    features = layers.Dense(feature_dim, activation='relu')(x)

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
    sum_exp = tf.reduce_sum(exp_similarity, axis=1)
    pos_exp = tf.reduce_sum(exp_similarity * labels_equal, axis=1)

    # 计算损失
    loss = -tf.math.log(pos_exp / sum_exp)
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


def train_msupcl_model(model, paired_train_generator, epochs=10, temperature=0.07):
    optimizer = optimizers.Adam(learning_rate=1e-4)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss_avg = tf.keras.metrics.Mean()
        for step in range(len(paired_train_generator)):
            (X1_batch, X2_batch), y_batch = paired_train_generator[step]
            with tf.GradientTape() as tape:
                features1 = model(X1_batch, training=True)
                features2 = model(X2_batch, training=True)
                # Concatenate features
                features = tf.concat([features1, features2], axis=0)

                # Concatenate label
                labels = tf.concat([y_batch, y_batch], axis=0)

                # Compute loss
                loss = supervised_contrastive_loss(labels, features, temperature)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss_avg.update_state(loss)
        print(f"Training Loss: {epoch_loss_avg.result():.4f}")


def linear_evaluation(model, train_generator, test_generator1,test_generator2, num_classes=2):
    # Freeze the base model
    for layer in model.layers:
        layer.trainable = False

    # Add a classification head
    features = model.output
    outputs = layers.Dense(num_classes, activation='softmax')(features)
    classifier_model = models.Model(inputs=model.input, outputs=outputs)

    # Compile the classifier
    classifier_model.compile(
        loss=losses.SparseCategoricalCrossentropy(),
        optimizer=optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )

    # Train the classifier on the combined dataset
    classifier_model.fit(
        train_generator,
        validation_data=test_generator1,  # or any validation generator
        epochs=5
    )

    # Evaluate on test sets
    print("Evaluating on Violence Test Set:")
    results_violence = classifier_model.evaluate(test_generator1)
    print(f"Violence Test Loss: {results_violence[0]}, Test Accuracy: {results_violence[1]}")

    print("Evaluating on TikTok Test Set:")
    results_tiktok = classifier_model.evaluate(test_generator2)
    print(f"TikTok Test Loss: {results_tiktok[0]}, Test Accuracy: {results_tiktok[1]}")

    return results_violence, results_tiktok


def nt_xent_loss(z_i, z_j, temperature=0.5):
    # z_i 和 z_j 的形状为 (batch_size, feature_dim)
    batch_size = tf.shape(z_i)[0]
    z = tf.concat([z_i, z_j], axis=0)  # (2*batch_size, feature_dim)

    # 计算相似度矩阵
    sim_matrix = tf.matmul(z, z, transpose_b=True)  # (2*batch_size, 2*batch_size)
    sim_matrix = sim_matrix / temperature

    # 创建标签
    labels = tf.range(batch_size)
    labels = tf.concat([labels, labels], axis=0)  # (2*batch_size,)

    # 创建掩码，排除自身对比
    mask = tf.eye(2 * batch_size)

    # 计算对比损失
    sim_matrix_exp = tf.exp(sim_matrix) * (1 - mask)
    sim_sum = tf.reduce_sum(sim_matrix_exp, axis=1, keepdims=True)
    pos_sim = tf.exp(tf.reduce_sum(z_i * z_j, axis=1) / temperature)

    loss = -tf.math.log(pos_sim / sim_sum[:batch_size, 0])
    loss = tf.reduce_mean(loss)
    return loss

def load_c3d_sscl_model(input_shape=(16, 112, 112, 3), feature_dim=512):
    # 定义输入
    inputs = layers.Input(shape=input_shape)

    # 第一层卷积和池化
    x = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)

    # 第二层卷积和池化
    x = layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

    # 第三、四层卷积和池化
    x = layers.Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

    # 第五、六层卷积和池化
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

    # 第七、八层卷积和池化
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

    # 展平和全连接层
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # 输出特征向量
    features = layers.Dense(feature_dim, activation='relu')(x)

    projections = layers.Dense(feature_dim)(features)

    # 构建模型
    model = models.Model(inputs=inputs, outputs=projections)

    return model

def train_simclr_model(model, train_generator, epochs=10, temperature=0.5):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss_avg = tf.keras.metrics.Mean()

        for step in range(len(train_generator)):
            (x_i, x_j), _ = train_generator[step]

            with tf.GradientTape() as tape:
                # 前向传播
                z_i = model(x_i, training=True)
                z_j = model(x_j, training=True)

                # 计算 NT-Xent 损失
                loss = nt_xent_loss(z_i, z_j, temperature)

            # 反向传播和优化
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            epoch_loss_avg.update_state(loss)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss_avg.result():.4f}")

def linear_evaluation_sscl(model, train_generator, val_generator,test_generator, num_classes=2):
    # 冻结编码器参数
    for layer in model.layers:
        layer.trainable = False

    # 添加线性分类器
    inputs = model.input
    features = model.get_layer(index=-2).output  # 获取特征层的输出
    outputs = layers.Dense(num_classes, activation='softmax')(features)
    classifier_model = models.Model(inputs=inputs, outputs=outputs)

    # 编译模型
    classifier_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )

    # 训练分类器
    classifier_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=5
    )

    # 评估模型
    results = classifier_model.evaluate(test_generator)
    print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")
    return results