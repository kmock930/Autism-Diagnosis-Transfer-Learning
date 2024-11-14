from tensorflow.keras import layers, models


def r2plus1d_block(input_tensor, filters, strides=(1, 1, 1)):
    # 2D 空间卷积
    x = layers.Conv3D(filters, kernel_size=(1, 3, 3), strides=strides, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 1D 时间卷积
    x = layers.Conv3D(filters, kernel_size=(3, 1, 1), strides=(1, 1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x


def r2plus1d_residual_block(input_tensor, filters, strides=(1, 1, 1)):
    x = r2plus1d_block(input_tensor, filters, strides)
    x = r2plus1d_block(x, filters)

    # 跳跃连接
    shortcut = input_tensor
    if strides != (1, 1, 1) or input_tensor.shape[-1] != filters:
        shortcut = layers.Conv3D(filters, kernel_size=1, strides=strides, padding='same', use_bias=False)(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x


def build_r2plus1d_model(input_shape=(16, 112, 112, 3), num_classes=2, feature_dim=512, include_top=False):
    inputs = layers.Input(shape=input_shape)

    # 初始卷积和池化层
    x = layers.Conv3D(64, kernel_size=(1, 7, 7), strides=(1, 2, 2), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(x)

    # 残差块
    x = r2plus1d_residual_block(x, 64)
    x = r2plus1d_residual_block(x, 64)

    x = r2plus1d_residual_block(x, 128, strides=(2, 2, 2))
    x = r2plus1d_residual_block(x, 128)

    x = r2plus1d_residual_block(x, 256, strides=(2, 2, 2))
    x = r2plus1d_residual_block(x, 256)

    x = r2plus1d_residual_block(x, 512, strides=(2, 2, 2))
    x = r2plus1d_residual_block(x, 512)

    # 全局平均池化
    x = layers.GlobalAveragePooling3D()(x)

    # 特征投影头（用于对比学习）
    features = layers.Dense(feature_dim, activation='relu')(x)

    if include_top:
        # 分类层（如果需要）
        outputs = layers.Dense(num_classes, activation='softmax')(features)
        model = models.Model(inputs, outputs)
    else:
        model = models.Model(inputs, features)

    return model


def load_r2plus1d_model(input_shape=(16, 112, 112, 3), feature_dim=512, include_top=False):
    model = build_r2plus1d_model(input_shape=input_shape, feature_dim=feature_dim, include_top=include_top)
    return model


def build_sscl_r2plus1d_model(input_shape=(16, 112, 112, 3), num_classes=2, feature_dim=512, include_top=False):
    inputs = layers.Input(shape=input_shape)

    # 初始卷积和池化层
    x = layers.Conv3D(64, kernel_size=(1, 7, 7), strides=(1, 2, 2), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(x)

    # 残差块
    x = r2plus1d_residual_block(x, 64)
    x = r2plus1d_residual_block(x, 64)

    x = r2plus1d_residual_block(x, 128, strides=(2, 2, 2))
    x = r2plus1d_residual_block(x, 128)

    x = r2plus1d_residual_block(x, 256, strides=(2, 2, 2))
    x = r2plus1d_residual_block(x, 256)

    x = r2plus1d_residual_block(x, 512, strides=(2, 2, 2))
    x = r2plus1d_residual_block(x, 512)

    # 全局平均池化
    x = layers.GlobalAveragePooling3D()(x)

    # 特征投影头（用于对比学习）
    features = layers.Dense(feature_dim, activation='relu')(x)

    projections = layers.Dense(feature_dim)(features)

    if include_top:
        # 分类层（如果需要）
        outputs = layers.Dense(num_classes, activation='softmax')(projections)
        model = models.Model(inputs, outputs)
    else:
        model = models.Model(inputs, projections)

    return model

def load_sscl_r2plus1d_model(input_shape=(16, 112, 112, 3), feature_dim=512, include_top=False):
    model = build_sscl_r2plus1d_model(input_shape=input_shape, feature_dim=feature_dim, include_top=include_top)
    return model