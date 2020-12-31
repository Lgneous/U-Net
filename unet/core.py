from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Conv2D


def _conv2d(
    filters, kernel_size=3, activation="relu", kernel_initializer="he_normal", **kwargs
):
    return Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        **kwargs,
    )


def contracting_block(inputs: tf.Tensor, channels: Optional[int] = None) -> tf.Tensor:
    """Contracting convolution block that consists of 2 successive 3x3 unpadded convolutions.

    :param inputs: Tensor of shape (N, H, W, C)
    :param channels: Number of channels in the output tensor, default to double the number of channels
               in the input tensor
    :returns: Tensor of shape (N, H-4, W-4, channels)
    """
    *_, c = inputs.shape
    if channels is None:
        channels = c * 2
    if channels <= c:
        raise ValueError(
            "Expected output channels to be greater than {}, found {}.".format(
                c, channels
            )
        )
    conv1 = _conv2d(channels)
    conv2 = _conv2d(channels)
    return conv1(conv2(inputs))


def expanding_block(inputs: tf.Tensor, channels: Optional[int] = None) -> tf.Tensor:
    """Expanding convolution block that consists of 2 successive 3x3 unpadded convolution.

    :param inputs: Tensor of shape (N, H, W, C)
    :param channels: Number of channels in the output tensor, default to half the number of channels
               in the input tensor
    :returns: Tensor of shape (N, H-4, W-4, channels)
    """
    *_, c = inputs.shape
    if channels is None:
        channels = c // 2
    if channels >= c:
        raise ValueError(
            "Expected output channels to be less than {}, found {}.".format(c, channels)
        )
    conv1 = _conv2d(channels)
    conv2 = _conv2d(channels)
    return conv1(conv2(inputs))
