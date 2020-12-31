import tensorflow as tf

import unet.core


class UNet(tf.keras.layers.Layer):
    def __init__(self, depth=4, units=None):
        super().__init__()
        self.depth = depth
        self.units = units

    def build(self, input_shape):
        s = 8
        for _ in range(self.depth):
            s *= 2
            s += 4
        if s > input_shape[1] or s > input_shape[2]:
            raise ValueError(
                "Image size must be greater than {}, found ({},{}).".format(
                    s, input_shape[1], input_shape[2]
                )
            )

    def __call__(self, inputs):
        super().__call__(inputs)
        block = unet.core.contracting_block(inputs, channels=64)
        contracting = [block]
        for _ in range(self.depth):
            block = tf.keras.layers.MaxPool2D(strides=(2, 2))(block)
            block = unet.core.contracting_block(block)
            contracting.append(block)
        block = contracting.pop()
        block = tf.keras.layers.Dropout(0.5)(block)
        for path in reversed(contracting):
            *_, channels = block.shape
            block = tf.keras.layers.Conv2DTranspose(
                channels // 2, 2, padding="same", strides=(2, 2)
            )(block)
            _, h, w, _ = block.shape
            path = unet.core.centered_crop(path, (h, w))
            block = tf.concat([path, block], axis=3)
            block = unet.core.expanding_block(block)
        if self.units is None:
            return block
        return tf.keras.layers.Conv2D(self.units, 1)(block)
