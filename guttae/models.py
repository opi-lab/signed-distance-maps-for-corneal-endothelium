from . import deeptrack as dt
from tensorflow.keras import layers

from tensorflow.keras.initializers import RandomNormal


def get_model(breadth, depth):
    """Creates a u-net generator that
    * Uses concatenation skip steps in the encoder
    * Uses maxpooling for downsampling
    * Uses resnet block for the base block
    * Uses instance normalization and leaky relu.
    Parameters
    ----------
    breadth : int
        Number of features in the top level. Each sequential level of the u-net
        increases the number of features by a factor of two.
    depth : int
        Number of levels to the u-net. If `n`, then there will be `n-1` pooling layers.
    """

    activation = layers.LeakyReLU(0.1)

    convolution_block = dt.layers.ResidualBlock(
        kernel_size=(3, 3),
        activation=activation,
        instance_norm=True,
    )

    return dt.models.unet(
        input_shape=(None, None, 1),
        conv_layers_dimensions=list(
            breadth * 2 ** n for n in range(depth - 1)
        ),
        base_conv_layers_dimensions=(breadth * 2 ** (depth - 1),) * 2,
        output_conv_layers_dimensions=[breadth, breadth],
        steps_per_pooling=2,
        number_of_outputs=1,
        output_kernel_size=1,
        output_activation="linear",
        downsampling_skip=False,
        encoder_convolution_block=convolution_block,
        decoder_convolution_block=convolution_block,
        base_convolution_block=convolution_block,
        output_convolution_block=convolution_block,
    )
