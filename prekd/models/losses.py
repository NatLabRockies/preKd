from nfp.frameworks import tf


# Above and blow 1.5 and -1.5 are more analytical errors, limitations of the instrument
# Try removing them from the MAE and try relabeling as categorical
def _inside_mask(y_true, cutoff):
    return tf.logical_and(
        tf.greater_equal(y_true, -cutoff),
        tf.less_equal(y_true, cutoff)
    )


def _safe_mean(values):
    values = tf.cast(values, tf.float32)
    count = tf.size(values)
    return tf.cond(
        count > 0,
        lambda: tf.reduce_mean(values),
        lambda: tf.constant(0.0, dtype=tf.float32),
    )


def hybrid_mae_bce_loss(y_true, y_pred, cutoff=1.5, bce_weight=0.5):
    """

    Custom loss function that applies:
    - MAE loss for values between -cutoff and cutoff
    - Binary Cross Entropy loss for values outside that range

    This gives more weight to the tails of the distribution and penalizes
    errors outside the cutoff more aggressively.

    Args:
        cutoff (float): The threshold value (default: 1.5)

    Returns:
        loss_fn: A TensorFlow loss function
    """
    inside_mask = _inside_mask(y_true, cutoff)

    mae_loss = _safe_mean(
        tf.abs(
            tf.boolean_mask(y_true, inside_mask)
            - tf.boolean_mask(y_pred, inside_mask)
        )
    )

    # Convert to a binary proxy target: inside cutoff -> 1, outside -> 0.
    binary_true = tf.cast(inside_mask, tf.float32)
    adjusted_pred = tf.sigmoid(-(tf.abs(y_pred) - cutoff))
    bce_loss = _safe_mean(
        tf.keras.losses.binary_crossentropy(binary_true, adjusted_pred)
    )

    bce_weight = tf.cast(bce_weight, tf.float32)
    return (1.0 - bce_weight) * mae_loss + bce_weight * bce_loss


class WeightedHybridMaeBceLoss(tf.keras.losses.Loss):
    def __init__(self, cutoff=1.5, bce_weight=0.5, name="hybrid_mae_bce_loss"):
        super().__init__(name=name)
        self.cutoff = cutoff
        self.bce_weight = bce_weight

    def call(self, y_true, y_pred):
        return hybrid_mae_bce_loss(
            y_true,
            y_pred,
            cutoff=self.cutoff,
            bce_weight=self.bce_weight,
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "cutoff": self.cutoff,
            "bce_weight": self.bce_weight,
        })
        return config


def mae_loss_cutoff(y_true, y_pred, cutoff=1.5):
    inside_mask = _inside_mask(y_true, cutoff)
    return _safe_mean(
        tf.abs(
            tf.boolean_mask(y_true, inside_mask)
            - tf.boolean_mask(y_pred, inside_mask)
        )
    )


def bce_loss_cutoff(y_true, y_pred, cutoff=1.5):
    inside_mask = _inside_mask(y_true, cutoff)
    binary_true = tf.cast(inside_mask, tf.float32)
    adjusted_pred = tf.sigmoid(-(tf.abs(y_pred) - cutoff))
    return _safe_mean(tf.keras.losses.binary_crossentropy(binary_true, adjusted_pred))
