from nfp.frameworks import tf


# Above and blow 1.5 and -1.5 are more analytical errors, limitations of the instrument
# Try removing them from the MAE and try relabeling as categorical
def hybrid_mae_bce_loss(y_true, y_pred, cutoff=1.5):
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
    # Create masks for values inside and outside the cutoff
    inside_mask = tf.logical_and(
        tf.greater_equal(y_true, -cutoff),
        tf.less_equal(y_true, cutoff)
    )
    outside_mask = tf.logical_not(inside_mask)

    # Handle case when all values are outside cutoff
    mae_loss = 0.0
    if tf.reduce_sum(tf.cast(inside_mask, tf.float32)) > 0:
        # For values inside the cutoff, use standard MAE
        mae_loss = tf.keras.losses.mean_absolute_error(
            tf.boolean_mask(y_true, inside_mask),
            tf.boolean_mask(y_pred, inside_mask)
        )

    # For values outside the cutoff, transform to binary classification problem
    # Calculate BCE loss for outliers (outside the cutoff)
    outside_true = tf.boolean_mask(y_true, outside_mask)
    outside_pred = tf.boolean_mask(y_pred, outside_mask)

    # Handle case when all values are inside cutoff
    bce_loss = 0.0
    if tf.reduce_sum(tf.cast(outside_mask, tf.float32)) > 0:
        # Apply binary cross-entropy for all values
        # Convert to binary target: 1 for values inside cutoff range, 0 for values outside
        binary_true = tf.cast(inside_mask, tf.float32)
        
        # Convert predictions to probabilities using sigmoid
        # Adjusting sigmoid to center properly around cutoff
        # Values inside cutoff range will be close to 1, outside cutoff close to 0
        adjusted_pred = tf.sigmoid(-(tf.abs(y_pred) - cutoff))
        
        # Calculate BCE loss
        bce_loss = tf.keras.losses.binary_crossentropy(
            binary_true,
            adjusted_pred
        )
        
    # Combine the losses - can adjust weights if needed
    #inside_count = tf.reduce_sum(tf.cast(inside_mask, tf.float32))
    #outside_count = tf.reduce_sum(tf.cast(outside_mask, tf.float32))
    #total_count = inside_count + outside_count
    
    ## Weight the losses based on proportion of samples
    #combined_loss = (inside_count / total_count) * mae_loss + (outside_count / total_count) * bce_loss
    combined_loss = mae_loss + bce_loss

    return combined_loss


def mae_loss_cutoff(y_true, y_pred, cutoff=1.5):
    # Create masks for values inside and outside the cutoff
    inside_mask = tf.logical_and(
        tf.greater_equal(y_true, -cutoff),
        tf.less_equal(y_true, cutoff)
    )
    outside_mask = tf.logical_not(inside_mask)

    # Handle case when all values are outside cutoff
    mae_loss = 0.0
    if tf.reduce_sum(tf.cast(inside_mask, tf.float32)) > 0:
        # For values inside the cutoff, use standard MAE
        mae_loss = tf.keras.losses.mean_absolute_error(
            tf.boolean_mask(y_true, inside_mask),
            tf.boolean_mask(y_pred, inside_mask)
        )
    return mae_loss


def bce_loss_cutoff(y_true, y_pred, cutoff=1.5):
    # Create masks for values inside and outside the cutoff
    inside_mask = tf.logical_and(
        tf.greater_equal(y_true, -cutoff),
        tf.less_equal(y_true, cutoff)
    )
    outside_mask = tf.logical_not(inside_mask)

    # For values outside the cutoff, transform to binary classification problem
    # Calculate BCE loss for outliers (outside the cutoff)
    outside_true = tf.boolean_mask(y_true, outside_mask)
    outside_pred = tf.boolean_mask(y_pred, outside_mask)

    # Handle case when all values are inside cutoff
    bce_loss = 0.0
    if tf.reduce_sum(tf.cast(outside_mask, tf.float32)) > 0:
        # Apply binary cross-entropy for all values
        # Convert to binary target: 1 for values inside cutoff range, 0 for values outside
        binary_true = tf.cast(inside_mask, tf.float32)
        
        # Convert predictions to probabilities using sigmoid
        # Adjusting sigmoid to center properly around cutoff
        # Values inside cutoff range will be close to 1, outside cutoff close to 0
        adjusted_pred = tf.sigmoid(-(tf.abs(y_pred) - cutoff))
        
        # Calculate BCE loss
        bce_loss = tf.keras.losses.binary_crossentropy(
            binary_true,
            adjusted_pred
        )
    return bce_loss
