from tensorflow.keras import layers, models

def r2plus1d_block(input_tensor, filters, strides=(1, 1, 1)):
    # 2D spatial convolution
    x = layers.Conv3D(filters, kernel_size=(1, 3, 3), strides=strides, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 1D temporal convolution
    x = layers.Conv3D(filters, kernel_size=(3, 1, 1), strides=(1, 1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x

def r2plus1d_residual_block(input_tensor, filters, strides=(1, 1, 1)):
    x = r2plus1d_block(input_tensor, filters, strides)
    x = r2plus1d_block(x, filters)

    # Skip connection
    shortcut = input_tensor
    if strides != (1, 1, 1) or input_tensor.shape[-1] != filters:
        shortcut = layers.Conv3D(filters, kernel_size=1, strides=strides, padding='same', use_bias=False)(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x

def build_r2plus1d_model(input_shape=(12, 64, 64, 3), num_classes=2, feature_dim=512, include_top=False):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution and pooling layers
    x = layers.Conv3D(64, kernel_size=(1, 7, 7), strides=(1, 2, 2), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(x)

    # Residual blocks
    x = r2plus1d_residual_block(x, 64)
    x = r2plus1d_residual_block(x, 64)

    x = r2plus1d_residual_block(x, 128, strides=(2, 2, 2))
    x = r2plus1d_residual_block(x, 128)

    x = r2plus1d_residual_block(x, 256, strides=(2, 2, 2))
    x = r2plus1d_residual_block(x, 256)

    x = r2plus1d_residual_block(x, 512, strides=(2, 2, 2))
    x = r2plus1d_residual_block(x, 512)

    # Global average pooling
    x = layers.GlobalAveragePooling3D()(x)

    # Feature projection head (for contrastive learning)
    features = layers.Dense(feature_dim, activation='relu')(x)

    if include_top:
        # Classification layer (if needed)
        outputs = layers.Dense(num_classes, activation='softmax')(features)
        model = models.Model(inputs, outputs)
    else:
        model = models.Model(inputs, features)

    return model

def load_r2plus1d_model(input_shape=(12, 64, 64, 3), feature_dim=512, include_top=False):
    model = build_r2plus1d_model(input_shape=input_shape, feature_dim=feature_dim, include_top=include_top)
    return model

def build_sscl_r2plus1d_model(input_shape=(12, 64, 64, 3), num_classes=2, feature_dim=512, include_top=False):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution and pooling layers
    x = layers.Conv3D(64, kernel_size=(1, 7, 7), strides=(1, 2, 2), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(x)

    # Residual blocks
    x = r2plus1d_residual_block(x, 64)
    x = r2plus1d_residual_block(x, 64)

    x = r2plus1d_residual_block(x, 128, strides=(2, 2, 2))
    x = r2plus1d_residual_block(x, 128)

    x = r2plus1d_residual_block(x, 256, strides=(2, 2, 2))
    x = r2plus1d_residual_block(x, 256)

    x = r2plus1d_residual_block(x, 512, strides=(2, 2, 2))
    x = r2plus1d_residual_block(x, 512)

    # Global average pooling
    x = layers.GlobalAveragePooling3D()(x)

    # Feature projection head (for contrastive learning)
    features = layers.Dense(feature_dim, activation='relu', name='features')(x)

    projections = layers.Dense(feature_dim, name='projections')(features)

    if include_top:
        # Classification layer (if needed)
        outputs = layers.Dense(num_classes, activation='softmax', name='classifier')(projections)
        model = models.Model(inputs, outputs)
    else:
        model = models.Model(inputs, outputs=[features, projections])

    return model

def load_sscl_r2plus1d_model(input_shape=(12, 64, 64, 3), feature_dim=512, include_top=False):
    model = build_sscl_r2plus1d_model(input_shape=input_shape, feature_dim=feature_dim, include_top=include_top)
    return model
