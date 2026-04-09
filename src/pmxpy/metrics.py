import numpy as np


def mpe(observed, predicted, relative=True):
    """
    Calculate Mean Performance Error.

    Parameters:
    -----------
    observed : array-like
        Observed values
    predicted : array-like
        Predicted values

    Returns:
    --------
    float : Mean Performance error
    """
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)
    pe = observed - predicted
    if relative:
        return 100 * np.mean(pe / predicted)
    else:
        return np.mean(pe)


def mdpe(observed, predicted, relative=True):
    """
    Calculate Median Performance Error.

    Parameters:
    -----------
    observed : array-like
        Observed values
    predicted : array-like
        Predicted values

    Returns:
    --------
    float : Median Performance error
    """
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)
    pe = observed - predicted
    if relative:
        return 100 * np.median(pe / predicted)
    else:
        return np.median(pe)


def mape(observed, predicted, relative=True):
    """
    Calculate Mean Absolute Performance Error (MAE / MAPE).

    Parameters:
    -----------
    observed : array-like
        Observed values
    predicted : array-like
        Predicted values

    Returns:
    --------
    float : Mean absolute Performance error
    """
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)
    pe = observed - predicted
    if relative:
        return 100 * np.mean(np.abs(pe / predicted))
    else:
        return np.mean(np.abs(pe))


def mdape(observed, predicted, relative=True):
    """
    Calculate Median Absolute Performance Error.

    Parameters:
    -----------
    observed : array-like
        Observed values
    predicted : array-like
        Predicted values

    Returns:
    --------
    float : Median absolute Performance error
    """
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)
    pe = observed - predicted
    if relative:
        return 100 * np.median(np.abs(pe / predicted))
    else:
        return np.median(np.abs(pe))


def wobble(observed, predicted, relative=True):
    """
    Wobble (Intra-individual variability).
    """
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)
    pe = observed - predicted
    mdpe_value = mdpe(observed, predicted, relative=relative)
    if relative:
        return np.median(np.abs(100 * pe / predicted - mdpe_value))
    else:
        return np.median(np.abs(pe - mdpe_value))


def divergence(observed, predicted, time, relative=True):
    """
    Divergence: slope of |PE| over time.

    Parameters:
    -----------
    observed : array-like
        Observed values
    predicted : array-like
        Predicted values
    time : array-like
        Time points (in minutes)

    Returns:
    --------
    float : Divergence in %/time (if relative=True) or units/time
    """
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)
    time = np.asarray(time)

    pe = observed - predicted
    if relative:
        abs_pe = np.abs(100 * pe / predicted)
    else:
        abs_pe = np.abs(pe)
    slope, _ = np.polyfit(time, abs_pe, 1)
    return slope
