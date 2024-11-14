import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics


def load_c3d_model(input_shape=(12, 64, 64, 3), feature_dim=512):
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
    x = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)

    # 第五、六层卷积和池化
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

    # 第七、八层卷积和池化
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)

    # 展平和全连接层
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # 输出特征向量
    features = layers.Dense(feature_dim, activation='relu')(x)

    # 构建模型
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss_avg = tf.keras.metrics.Mean()
        for step in range(len(train_generator)):
            X_batch, y_batch, dataset_ids = train_generator[step]
            with tf.GradientTape() as tape:
                features = model(X_batch, training=True)
                loss = supervised_contrastive_loss(y_batch, features, dataset_ids, temperature)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss_avg.update_state(loss)
        print(f"Training Loss: {epoch_loss_avg.result():.4f}")


def linear_evaluation(model, train_generator,test_generator1,test_generator2, num_classes=2, num_epochs=5):
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
        epochs=num_epochs
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
    batch_size = tf.shape(z_i)[0]

    # 归一化特征向量
    z_i = tf.math.l2_normalize(z_i, axis=1)
    z_j = tf.math.l2_normalize(z_j, axis=1)

    # 拼接特征向量
    z = tf.concat([z_i, z_j], axis=0)  # (2*batch_size, feature_dim)

    # 计算相似度矩阵
    similarity_matrix = tf.matmul(z, z, transpose_b=True)  # (2*batch_size, 2*batch_size)
    similarity_matrix = similarity_matrix / temperature

    # 创建掩码，排除自身对比
    mask = tf.eye(2 * batch_size)
    logits = similarity_matrix - mask * 1e9  # 将自身相似度设为一个极小值

    # 创建标签：对于每个样本，其正样本索引为 (i + batch_size) % (2 * batch_size)
    positive_indices = tf.concat([tf.range(batch_size, 2 * batch_size), tf.range(0, batch_size)], axis=0)
    labels = positive_indices

    # 计算交叉熵损失
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(loss)
    return loss

def load_c3d_sscl_model(input_shape=(12, 64, 64, 3), feature_dim=512):
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
    x = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)

    # 第五、六层卷积和池化
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

    # 第七、八层卷积和池化
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)

    # 展平和全连接层
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # 输出特征向量
    features = layers.Dense(feature_dim, activation='relu', name='features')(x)

    projections = layers.Dense(feature_dim, name='projections')(features)

    # 构建模型
    model = models.Model(inputs=inputs, outputs=[features, projections])

    return model

def train_simclr_model(model, train_generator, epochs=10, temperature=0.5):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss_avg = tf.keras.metrics.Mean()
        for step in range(len(train_generator)):
            (x_i, x_j), _ = train_generator[step]
            with tf.GradientTape() as tape:
                # 获取投影后的特征
                _, z_i = model(x_i, training=True)
                _, z_j = model(x_j, training=True)
                # 计算 NT-Xent 损失
                loss = nt_xent_loss(z_i, z_j, temperature)
            # 反向传播和优化
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss_avg.update_state(loss)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss_avg.result():.4f}")

def linear_evaluation_sscl(model, train_generator, val_generator, test_generator, num_classes=2,num_epochs=3):
    # 冻结编码器参数
    for layer in model.layers:
        layer.trainable = False
    # 定义输入为模型的输入
    inputs = model.input
    # 使用features作为特征
    features = model.outputs[0]
    # 添加分类层
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
        epochs=num_epochs
    )
    # 评估模型
    results = classifier_model.evaluate(test_generator)
    print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")
    return results