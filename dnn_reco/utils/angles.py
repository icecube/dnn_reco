from __future__ import division, print_function
import numpy as np
import tensorflow as tf


def get_angle_deviation(azimuth1, zenith1, azimuth2, zenith2):
    """Get opening angle of two vectors defined by (azimuth, zenith)

    Parameters
    ----------
    azimuth1 : numpy.ndarray or float
        Azimuth of vector 1 in rad.
    zenith1 : numpy.ndarray or float
        Zenith of vector 1 in rad.
    azimuth2 : numpy.ndarray or float
        Azimuth of vector 2 in rad.
    zenith2 : numpy.ndarray or float
        Zenith of vector 2 in rad.

    Returns
    -------
    numpy.ndarray or float
        The opening angle in rad between the vector 1 and 2.
        Same shape as input vectors.
    """
    cos_dist = np.cos(azimuth1 - azimuth2) * np.sin(zenith1) * np.sin(
        zenith2
    ) + np.cos(zenith1) * np.cos(zenith2)
    cos_dist = np.clip(cos_dist, -1.0, 1.0)
    return np.arccos(cos_dist)


def tf_get_angle_deviation(azimuth1, zenith1, azimuth2, zenith2):
    """Get opening angle of two vectors defined by (azimuth, zenith)

    Parameters
    ----------
    azimuth1 : numpy.ndarray or float
        Azimuth of vector 1 in rad.
    zenith1 : numpy.ndarray or float
        Zenith of vector 1 in rad.
    azimuth2 : numpy.ndarray or float
        Azimuth of vector 2 in rad.
    zenith2 : numpy.ndarray or float
        Zenith of vector 2 in rad.

    Returns
    -------
    numpy.ndarray or float
        The opening angle in rad between the vector 1 and 2.
        Same shape as input vectors.
    """
    cos_dist = tf.cos(azimuth1 - azimuth2) * tf.sin(zenith1) * tf.sin(
        zenith2
    ) + tf.cos(zenith1) * tf.cos(zenith2)
    cos_dist = tf.clip_by_value(cos_dist, -1.0, 1.0)
    return tf.acos(cos_dist)


def get_angle(vec1, vec2, dtype=np.float64):
    """Calculate opening angle between two direction vectors.

    vec1/2 : shape: [?,3] or [3]
    https://www.cs.berkeley.edu/~wkahan/Mindless.pdf p.47/56

    Parameters
    ----------
    vec1 : numpy.ndarray
        Direction vector 1.
        Shape: [?,3] or [3]
    vec2 : numpy.ndarray
        Direction vector 2.
        Shape: [?,3] or [3]
    dtype : numpy.dtype, optional
        The float precision.

    Returns
    -------
    numpy.ndarray
        The opening angle in rad between the direction vector 1 and 2.
        Same shape as input vectors.
    """
    # transform into numpy array with dtype
    vec1 = np.array(vec1, dtype=dtype)
    vec2 = np.array(vec2, dtype=dtype)

    assert vec1.shape[-1] == 3, "Expect shape [?,3] or [3], but got {}".format(
        vec1.shape
    )
    assert vec2.shape[-1] == 3, "Expect shape [?,3] or [3], but got {}".format(
        vec2.shape
    )

    norm1 = np.linalg.norm(vec1, axis=-1, keepdims=True)
    norm2 = np.linalg.norm(vec2, axis=-1, keepdims=True)
    tmp1 = vec1 * norm2
    tmp2 = vec2 * norm1

    tmp3 = np.linalg.norm(tmp1 - tmp2, axis=-1)
    tmp4 = np.linalg.norm(tmp1 + tmp2, axis=-1)

    theta = 2 * np.arctan2(tmp3, tmp4)

    return theta
