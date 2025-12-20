import numpy as np

def discretizing_activation_fn(smooth_preds, target_std):
    """
    Rounds values away from 0 to the nearest multiple of `interval`.
    
    'interval' is the what a basketball point is in terms of
    scaled units. Thus, after obtaining 'discrete_preds' using
    this activation function, unscaling will give integer
    values corresponding to actual plus-minus predictions.

    Parameters
    ----------
    smooth_preds : array-like
        Continuous predictions (scaled).
    target_std : float
        Standard deviation of original scale

    Returns
    -------
    discrete_preds : np.ndarray
        Values rounded away from zero to the given interval.
    """
    smooth_preds = np.asarray(smooth_preds)
    interval = 1 / target_std

    scaled = smooth_preds / interval

    discrete = np.where(
        scaled > 0,
        np.ceil(scaled),
        np.where(scaled < 0, np.floor(scaled), 0)
    )

    return discrete * interval



def discretizing_activation_fn2(smooth_preds, target_std):
    """
    Rounds smooth predictions according to a custom rule:
    
    - If the absolute value of the prediction is less than one interval,
      round away from 0.
    - Otherwise, round to the nearest interval.
    
    Parameters
    ----------
    smooth_preds : array-like
        Continuous predictions (scaled).
    target_std : float
        Standard deviation of original scale.
        
    Returns
    -------
    discrete_preds : np.ndarray
        Discretized values according to the rule.
    """
    smooth_preds = np.asarray(smooth_preds)
    interval = 1 / target_std

    # Compute scaled predictions in terms of intervals
    scaled = smooth_preds / interval

    # Apply rounding rules
    discrete = np.where(
        (scaled > 0) & (scaled < 1),
        1,  # round away from 0 for small positive
        np.where(
            (scaled < 0) & (scaled > -1),
            -1,  # round away from 0 for small negative
            np.round(scaled)  # otherwise, round to nearest interval
        )
    )

    return discrete * interval