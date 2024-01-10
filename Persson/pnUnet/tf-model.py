import tensorflow as tf
from tensorflow.keras import layers, Model

# This double convolution happens at each step in the figure
class DoubleConv(layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = tf.keras.Sequential([
            layers.Conv2D(out_channels, 3, 1, 'same', use_bias=False, input_shape=(None, None, in_channels)),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(out_channels, 3, 1, 'same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

    def call(self, x):
        return self.conv(x)

class PNUNET(Model):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(PNUNET, self).__init__()
        
        self.downs = []
        self.ups = []
        self.pool = layers.MaxPooling2D(pool_size=(2, 2), strides=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(layers.Conv2DTranspose(feature, kernel_size=2, strides=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = layers.Conv2D(out_channels, kernel_size=1)

    def call(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = tf.image.resize(x, skip_connection.shape[1:3])
            x = tf.concat([skip_connection, x], axis=-1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)

def test():
    x = tf.random.normal((3, 161, 161, 1))
    model = PNUNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()
