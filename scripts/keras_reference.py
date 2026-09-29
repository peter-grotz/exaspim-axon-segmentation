"""Regenerate ``tests/data/keras_reference_axon.npz`` with the original Keras model.

Run in a throwaway TensorFlow environment (TF is never a dependency of this package)::

    uv run --no-project --python 3.11 --with tensorflow==2.16.2 --with tf_keras==2.16.0 \
        --with "numpy<2" python scripts/keras_reference.py AXON_model_210409.hdf5 \
        parity_inputs.npz tests/data/keras_reference_axon.npz

``parity_inputs.npz`` holds ``real`` (a 64³ float16 patch of exaSPIM_851474 fused level 2 centered
at index (2800, 5113, 6751), divided by its 99.9th percentile and clipped to [0, 1]) and ``noise``
(``default_rng(0).random((128,) * 3) ** 8`` as float16).
"""

import os
import sys

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import tensorflow as tf
from tf_keras.layers import (
    BatchNormalization,
    Conv3D,
    Conv3DTranspose,
    Cropping3D,
    Input,
    MaxPooling3D,
    concatenate,
)
from tf_keras.models import Model


def get_net(n):
    """Capsule ``code/models/model.py`` architecture verbatim, input side parameterized."""
    input = Input((n, n, n, 1))
    conv1 = Conv3D(32, (3, 3, 3), activation="relu", padding="same")(input)
    batch1 = BatchNormalization()(conv1)
    conv1 = Conv3D(64, (3, 3, 3), activation="relu", padding="same")(batch1)
    batch1 = BatchNormalization()(conv1)
    pool2 = MaxPooling3D((2, 2, 2))(batch1)
    conv2 = Conv3D(64, (3, 3, 3), activation="relu", padding="same")(pool2)
    batch2 = BatchNormalization()(conv2)
    conv2 = Conv3D(128, (3, 3, 3), activation="relu", padding="same")(batch2)
    batch2 = BatchNormalization()(conv2)
    pool3 = MaxPooling3D((2, 2, 2))(batch2)
    conv3 = Conv3D(128, (3, 3, 3), activation="relu", padding="same")(pool3)
    batch3 = BatchNormalization()(conv3)
    conv3 = Conv3D(256, (3, 3, 3), activation="relu", padding="same")(batch3)
    batch3 = BatchNormalization()(conv3)
    pool4 = MaxPooling3D((2, 2, 2))(batch3)
    conv4 = Conv3D(256, (3, 3, 3), activation="relu", padding="same")(pool4)
    batch4 = BatchNormalization()(conv4)
    conv4 = Conv3D(512, (3, 3, 3), activation="relu", padding="same")(batch4)
    batch4 = BatchNormalization()(conv4)
    up5 = Conv3DTranspose(512, (2, 2, 2), strides=(2, 2, 2), padding="same", activation="relu")(
        batch4
    )
    merge5 = concatenate([up5, batch3])
    conv5 = Conv3D(256, (3, 3, 3), activation="relu")(merge5)
    batch5 = BatchNormalization()(conv5)
    conv5 = Conv3D(256, (3, 3, 3), activation="relu")(batch5)
    batch5 = BatchNormalization()(conv5)
    up6 = Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), activation="relu")(batch5)
    merge6 = concatenate([up6, Cropping3D(cropping=((4, 4), (4, 4), (4, 4)))(batch2)])
    conv6 = Conv3D(128, (3, 3, 3), activation="relu")(merge6)
    batch6 = BatchNormalization()(conv6)
    conv6 = Conv3D(128, (3, 3, 3), activation="relu")(batch6)
    batch6 = BatchNormalization()(conv6)
    up7 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding="same", activation="relu")(
        batch6
    )
    merge7 = concatenate([up7, Cropping3D(cropping=((12, 12), (12, 12), (12, 12)))(batch1)])
    conv7 = Conv3D(64, (3, 3, 3), activation="relu")(merge7)
    batch7 = BatchNormalization()(conv7)
    conv7 = Conv3D(64, (3, 3, 3), activation="relu")(batch7)
    batch7 = BatchNormalization()(conv7)
    preds = Conv3D(1, (1, 1, 1), activation="sigmoid")(batch7)
    return Model(inputs=input, outputs=preds)


weights, inputs, out = sys.argv[1:4]
x = np.load(inputs)
real = x["real"].astype(np.float32)
noise = (
    (np.random.default_rng(0).random((128, 128, 128)) ** 8).astype(np.float16).astype(np.float32)
)
assert np.array_equal(noise, x["noise"].astype(np.float32))
m64, m128 = get_net(64), get_net(128)
m64.load_weights(weights)
m128.load_weights(weights)
y_real = m64.predict(real[None, ..., None], verbose=0)[0, ..., 0]
y_noise64 = m64.predict(noise[:64, :64, :64][None, ..., None], verbose=0)[0, ..., 0]
y_noise128 = m128.predict(noise[None, ..., None], verbose=0)[0, ..., 0]
print(
    tf.__version__,
    y_real.shape,
    y_noise128.shape,
    float(y_real.min()),
    float(y_real.max()),
    float(y_real.mean()),
)
np.savez_compressed(
    out,
    real_input=x["real"],
    real_output=y_real,
    noise64_output=y_noise64,
    noise128_output_sub4=y_noise128[::4, ::4, ::4],
    tensorflow_version=tf.__version__,
    weights=os.path.basename(weights),
)
