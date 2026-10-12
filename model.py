"""Model definition, masked loss/metric and prediction helpers."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import keras  # Keras 3 directly; `from tensorflow import keras` can load the legacy tf_keras shim
from keras import layers

import data


@keras.saving.register_keras_serializable(package="accai")
def masked_mse(y_true, y_pred):
    """Weighted MSE over the predicted notes, ignoring notes with weight 0.

    y_true: (batch, PREDICTION_SIZE, 2) -> [..., 0] target, [..., 1] weight
    y_pred: (batch, PREDICTION_SIZE, 1)

    Divided by the number of used notes (not the sum of weights), so a map with weight 0.8
    (fewer replays) really counts 0.8 as much as a full-weight map.
    """
    target = y_true[..., 0]
    weight = y_true[..., 1]
    pred = y_pred[..., 0]
    used = tf.cast(weight > 0, weight.dtype)
    return tf.reduce_sum(weight * tf.square(target - pred), axis=-1) / tf.maximum(tf.reduce_sum(used, axis=-1), 1.0)


@keras.saving.register_keras_serializable(package="accai")
def masked_mae(y_true, y_pred):
    target = y_true[..., 0]
    used = tf.cast(y_true[..., 1] > 0, y_true.dtype)  # plain average error, not weighted
    pred = y_pred[..., 0]
    return tf.reduce_sum(used * tf.abs(target - pred), axis=-1) / tf.maximum(tf.reduce_sum(used, axis=-1), 1.0)


def build_model(note_size: int = data.NOTE_SIZE, lstm_units=(12, 10), dense_units: int = 16) -> keras.Model:
    """Same architecture as the notebook: 2 bidirectional LSTMs over the whole window, skip connection
    with the raw input, crop to the predicted notes, then a small per-note dense head.

    Layer names "input_1" and "time_distributed_2" are kept so the ONNX input/output names don't change.
    """
    inp = keras.Input(shape=(data.SEGMENT_SIZE, note_size), name="input_1")
    x = inp
    for units in lstm_units:
        x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
    x = layers.Concatenate()([x, inp])
    x = layers.Cropping1D(cropping=(data.PRE_SEGMENT_SIZE, data.POST_SEGMENT_SIZE))(x)
    x = layers.TimeDistributed(layers.Dense(dense_units, activation="relu"))(x)
    x = layers.TimeDistributed(layers.Dense(dense_units, activation="relu"))(x)
    out = layers.TimeDistributed(layers.Dense(1, activation="linear"), name="time_distributed_2")(x)
    return keras.Model(inp, out)


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    model.compile(
        # clipnorm stops the loss spikes seen in the old runs from wrecking the weights
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=masked_mse,
        metrics=[masked_mae],
    )
    return model


def keras_predictor(model: keras.Model, batch_size: int = 1024):
    def predict(X: np.ndarray) -> np.ndarray:
        return model.predict(X, batch_size=batch_size, verbose=0)
    return predict
