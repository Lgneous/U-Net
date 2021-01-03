from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D

__all__ = ["contracting_block", "expanding_block", "centered_crop"]


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


def contracting_block(
    inputs: tf.Tensor, channels: Optional[int] = None, returns_same_dims: bool = True
) -> tf.Tensor:
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
    conv1 = _conv2d(channels, padding="same" if returns_same_dims else "valid")
    conv2 = _conv2d(channels, padding="same" if returns_same_dims else "valid")
    return conv1(conv2(inputs))


def expanding_block(
    inputs: tf.Tensor, channels: Optional[int] = None, returns_same_dims: bool = True
) -> tf.Tensor:
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
    conv1 = _conv2d(channels, padding="same" if returns_same_dims else "valid")
    conv2 = _conv2d(channels, padding="same" if returns_same_dims else "valid")
    return conv1(conv2(inputs))


def centered_crop(inputs: tf.Tensor, shape: Tuple[int, int]) -> tf.Tensor:
    """Resize inputs to shape."""
    _, old_h, old_w, _ = inputs.shape
    h, w = shape
    diff_h = old_h - h
    diff_w = old_w - w
    offset_top = diff_h // 2
    offset_bottom = diff_h - offset_top
    offset_left = diff_w // 2
    offset_right = diff_w - offset_left
    return inputs[:, offset_top:-offset_bottom, offset_left:-offset_right, :]
