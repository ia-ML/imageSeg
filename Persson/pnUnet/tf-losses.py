import tensorflow as tf

class BinaryDiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1.0):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_pred = tf.nn.sigmoid(y_pred)
        # y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Apply threshold if needed

        # Flatten label and prediction tensors
        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.reshape(y_true, [-1])

        # Calculate Dice score
        intersection = tf.reduce_sum(y_pred * y_true)
        dice = (2. * intersection + self.smooth) / (tf.reduce_sum(y_pred) + tf.reduce_sum(y_true) + self.smooth)

        # Return Dice loss
        return 1 - dice

def test():
    # Example usage
    y_true = tf.constant([0, 1, 1, 0, 1], dtype=tf.float32)
    y_pred = tf.constant([0.1, 0.9, 0.8, 0.2, 0.3], dtype=tf.float32)
    loss = BinaryDiceLoss()
    print("Dice loss:", loss(y_true, y_pred))

if __name__ == "__main__":
    test()
