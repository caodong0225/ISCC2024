import tabnet_model
from process_data import train_ds, feature_names, cat_str_feature_names, cat_int_feature_names, cat_embed_dims
import tensorflow as tf
from params import tabnet_params


def create_keras_input_layer(feature_names, cat_str_feature_names, cat_int_feature_names):
    model_inputs = list()

    for name in feature_names:
        if name in cat_str_feature_names:
            dtype = tf.string
        elif name in cat_int_feature_names:
            dtype = tf.int64
        else:
            dtype = tf.float32

        shape = (1,) if dtype == tf.float32 else ()
        model_inputs.append(tf.keras.Input(shape=shape, name=name, dtype=dtype))

    return model_inputs


def encode_categorical_feature(keras_input, feature_name, dataset,
                               embed_dim, is_string):
    feature_ds = dataset.map(lambda x, _: x[feature_name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    lookup_fn = tf.keras.layers.StringLookup if is_string else tf.keras.layers.IntegerLookup
    lookup = lookup_fn(output_mode="int")
    lookup.adapt(feature_ds)
    encoded_feature = lookup(keras_input)
    embedded_feature = tf.keras.layers.Embedding(
        input_dim=lookup.vocabulary_size(),
        output_dim=embed_dim,
        name=f"{feature_name}_embedding"
    )(encoded_feature)

    return embedded_feature


def encode_features(keras_inputs, feature_names,
                    cat_str_feature_names, cat_int_feature_names, cat_embed_dims,
                    dataset):
    encoded_features = list()

    for keras_input, feature_name in zip(keras_inputs, feature_names):
        if feature_name in cat_str_feature_names or feature_name in cat_int_feature_names:
            # add embedding layer for all categorical features
            embed_dim = cat_embed_dims[feature_name] if feature_name in cat_embed_dims.keys() else 1
            encoded_features.append(
                encode_categorical_feature(keras_input, feature_name,
                                           dataset,
                                           embed_dim,
                                           feature_name in cat_str_feature_names)
            )
        else:
            # no encoding for numerical features
            encoded_features.append(keras_input)

    return encoded_features


def build_model():
    # Keras model using Functional API
    inputs = create_keras_input_layer(feature_names,
                                      cat_str_feature_names,
                                      cat_int_feature_names)
    x = encode_features(inputs, feature_names,
                        cat_str_feature_names, cat_int_feature_names, cat_embed_dims,
                        train_ds)
    x = tf.keras.layers.Concatenate()(x)
    x = tabnet_model.TabNetEncoder(**tabnet_params)(x)
    output = tf.keras.layers.Dense(12, activation='softmax', kernel_initializer='glorot_uniform')(x)
    model = tf.keras.Model(inputs, output)
    return model
