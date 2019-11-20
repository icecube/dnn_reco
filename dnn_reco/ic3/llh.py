#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np


class DNN_LLH(object):

    """A class for calculating the PDF obtained from the DNN reco for models
    that estimate the direction vector components and their uncertainty in
    independent 1D Gaussian Likelihoods, while the direction vector is
    normalized within the neural network model.

    Note: if the used neural network model does not fulfill these requirements,
    the PDF will be incorrect!

    Attributes
    ----------
    dir_x : float
        The best fit direction vector x component.
        This is the output of the DNN reco for the x-component.
    dir_x_s : np.array
        The x component of the sampled direction vectors.
    dir_y : float
        The best fit direction vector y component.
        This is the output of the DNN reco for the y-component.
    dir_y_s : np.array
        The y component of the sampled direction vectors.
    dir_z : float
        The best fit direction vector z component.
        This is the output of the DNN reco for the z-component.
    dir_z_s : np.array
        The z component of the sampled direction vectors.
    neg_llh_values : np.array
        The negative log probability values for the sampled direcctions.
    unc_x : float
        The estimated uncertainty for the direction vector x component.
        This is the output of the DNN reco for the estimated uncertainty.
    unc_y : float
        The estimated uncertainty for the direction vector y component.
        This is the output of the DNN reco for the estimated uncertainty.
    unc_z : float
        The estimated uncertainty for the direction vector z component.
        This is the output of the DNN reco for the estimated uncertainty.
    """

    def __init__(self, dir_x, dir_y, dir_z, unc_x, unc_y, unc_z,
                 num_samples=1000000, random_seed=42):
        """Initialize DNN LLH object.

        Parameters
        ----------
        dir_x : float
            The best fit direction vector x component.
            This is the output of the DNN reco for the x-component.
        dir_y : float
            The best fit direction vector y component.
            This is the output of the DNN reco for the y-component.
        dir_z : float
            The best fit direction vector z component.
            This is the output of the DNN reco for the z-component.
        unc_x : float
            The estimated uncertainty for the direction vector x component.
            This is the output of the DNN reco for the estimated uncertainty.
        unc_y : float
            The estimated uncertainty for the direction vector y component.
            This is the output of the DNN reco for the estimated uncertainty.
        unc_z : float
            The estimated uncertainty for the direction vector z component.
            This is the output of the DNN reco for the estimated uncertainty.
        num_samples : int, optional
            Number of samples to sample for internal calculations.
            The more samples, the more accurate, but also slower.
        random_seed : int, optional
            Random seed for sampling.
        """
        self.dir_x = dir_x
        self.dir_y = dir_y
        self.dir_z = dir_z
        self.unc_x = unc_x
        self.unc_y = unc_y
        self.unc_z = unc_z
        self._num_samples = num_samples
        self._random_seed = random_seed
        self._random_state = np.random.RandomState(self._random_seed)
        self.zenith, self.azimuth = self.get_zenith_azimuth(
                                        self.dir_x, self.dir_y, self.dir_z)
        # sample contours
        self.dir_x_s, self.dir_y_s, self.dir_z_s = \
            self.sample(self._num_samples)

        self.neg_llh_values = -self.log_prob_dir(
            self.dir_x_s, self.dir_y_s, self.dir_z_s)

        # sort sampled points according to neg llh
        sorted_indices = np.argsort(self.neg_llh_values)
        self.dir_x_s = self.dir_x_s[sorted_indices]
        self.dir_y_s = self.dir_y_s[sorted_indices]
        self.dir_z_s = self.dir_z_s[sorted_indices]
        self.neg_llh_values = self.neg_llh_values[sorted_indices]

        # get sampled zenith and azimuth
        self.zenith_s, self.azimuth_s = self.get_zenith_azimuth(
                        self.dir_x_s, self.dir_y_s, self.dir_z_s)

    def get_dir_vec(self, zenith, azimuth, with_flip=True):
        """Get direction vectors from zeniths and azimuths.

        Parameters
        ----------
        zenith : np.array
            The zenith angles in radians.
        azimuth : np.array
            The azimuth angles in radians.
        with_flip : bool, optional
            If True, then the direction vectors show in the opposite direction
            than the zenith/azimuth pairs. This is common within IceCube
            software, since the direction vector shows along the particle
            direction, whereas the zenith/azimuth shows to the source position.

        Returns
        -------
        np.array, np.array, np.array
            The direction vector components.
        """
        sin_zenith = np.sin(zenith)
        dir_x = sin_zenith * np.cos(azimuth)
        dir_y = sin_zenith * np.sin(azimuth)
        dir_z = np.cos(zenith)
        if with_flip:
            dir_x = -dir_x
            dir_y = -dir_y
            dir_z = -dir_z

        return dir_x, dir_y, dir_z

    def get_zenith_azimuth(self, dir_x, dir_y, dir_z,
                           with_flip=True):
        """Get zeniths and azimuths [in radians] from direction vector.

        Parameters
        ----------
        dir_x : np.array
            The direction vector x component.
        dir_y : np.array
            The direction vector y component.
        dir_z : np.array
            The direction vector z component.
        with_flip : bool, optional
            If True, then the direction vectors show in the opposite direction
            than the zenith/azimuth pairs. This is common within IceCube
            software, since the direction vector shows along the particle
            direction, whereas the zenith/azimuth shows to the source position.

        Returns
        -------
        np.array, np.array
            The zeniths and azimuth angles in radians.
        """
        # normalize
        dir_x, dir_y, dir_z = self.normalize_dir(dir_x, dir_y, dir_z)

        zenith = np.arccos(np.clip(-dir_z, -1, 1))
        azimuth = (np.arctan2(-dir_y, -dir_x) + 2 * np.pi) % (2 * np.pi)

        return zenith, azimuth

    def log_gauss(self, x, mu, sigma):
        """Calculate the log probability of Gaussian LLH.
        Note: the term here is proportional to the Gaussian LLH, but is not
        the full LLH.

        Parameters
        ----------
        x : np.array
            The estimated positions for which to calculate the llh.
        mu : float or np.array
            The mean of the Gaussian distribution-
        sigma : float np.array
            The std. devaition of the Gaussian distribution-

        Returns
        -------
        np.array
            The log probability of Gaussian LLH.
        """
        return -2*np.log(sigma) - ((x - mu) / sigma)**2

    def log_prob(self, zenith, azimuth):
        """Calculate the log probability for given zenith/azimuth pairs.

        Parameters
        ----------
        zenith : np.array
            The zenith angles in radians.
        azimuth : np.array
            The azimuth angles in radians.

        Returns
        -------
        np.array
            The log probability for given zenith/azimuth pairs.
        """
        dir_x, dir_y, dir_z = self.get_dir_vec(zenith, azimuth)
        return self.log_prob_dir(dir_x, dir_y, dir_z)

    def log_prob_dir(self, dir_x, dir_y, dir_z):
        """Calculate the log probability for given direction vectors.

        Parameters
        ----------
        dir_x : np.array
            The direction vector x component.
        dir_y : np.array
            The direction vector y component.
        dir_z : np.array
            The direction vector z component.

        Returns
        -------
        np.array
            The log probability for given zenith/azimuth pairs.
        """
        log_p = self.log_gauss(dir_x, self.dir_x, self.unc_x)
        log_p += self.log_gauss(dir_y, self.dir_y, self.unc_y)
        log_p += self.log_gauss(dir_z, self.dir_z, self.unc_z)
        return log_p

    def sample(self, n):
        """Sample direction vectors from the distribution

        Parameters
        ----------
        n : int
            Number of samples to generate.

        Returns
        -------
        np.array, np.array, np.array
            The sampled direction vector components.
        """
        dir_x_s = self._random_state.normal(self.dir_x, self.unc_x, n)
        dir_y_s = self._random_state.normal(self.dir_y, self.unc_y, n)
        dir_z_s = self._random_state.normal(self.dir_z, self.unc_z, n)
        dir_x_s, dir_y_s, dir_z_s = self.normalize_dir(
                                    dir_x_s, dir_y_s, dir_z_s)
        return dir_x_s, dir_y_s, dir_z_s

    def normalize_dir(self, dir_x, dir_y, dir_z):
        """Normalize a direction vector to a unit vector.

        Parameters
        ----------
        dir_x : np.array
            The direction vector x component.
        dir_y : np.array
            The direction vector y component.
        dir_z : np.array
            The direction vector z component.

        Returns
        -------
        np.array, np.array, np.array
            The normalized direction vector components.
        """
        norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
        return dir_x/norm, dir_y/norm, dir_z/norm

    def cdf(self, zenith, azimuth):
        """Calculate cumulative probability for given zenith/azimuth pairs.

        Parameters
        ----------
        zenith : np.array
            The zenith angles in radians.
        azimuth : np.array
            The azimuth angles in radians.

        Returns
        -------
        np.array
            The cumulative probabilty for the given zenith/azimuth pairs.
        """
        dir_x, dir_y, dir_z = self.get_dir_vec(zenith, azimuth)
        return self.cdf_dir(dir_x, dir_y, dir_z)

    def cdf_dir(self, dir_x, dir_y, dir_z):
        """Calculate cumulative probability for given direction vectors.

        Parameters
        ----------
        dir_x : np.array
            The direction vector x component.
        dir_y : np.array
            The direction vector y component.
        dir_z : np.array
            The direction vector z component.

        Returns
        -------
        np.array
            The cumulative probabilty for the given direction vectors.
        """
        dir_x, dir_y, dir_z = self.normalize_dir(dir_x, dir_y, dir_z)
        neg_llh = -self.log_prob_dir(dir_x, dir_y, dir_z)
        pos = np.searchsorted(self.neg_llh_values, neg_llh)
        cdf = 1.0*pos / self._num_samples
        return cdf

    def _get_level_indices(self, level=0.5, delta=0.001):
        """Get indices of sampled diections, which belong to the specified
        contour as defined by: level +- delta.

        Parameters
        ----------
        level : float, optional
            The contour level. Example: a level of 0.7 means that 70% of events
            are within this contour.
        delta : float, optional
            The contour is provided by selecting directions from the sampled
            ones which have cdf values within [level - delta, level + delta].
            The smaller delta, the more accurate the contour will be. However,
            the number of available sample points for the contour will also
            decrease.

        Returns
        -------
        int, int
            The starting and stopping index for a slice of sampled events
            that lie within the contour [level - delta, level + delta].

        Raises
        ------
        ValueError
            If number of resulting samples is too low.
        """
        assert level >= 0., level
        assert level <= 1., level

        index_at_level = int(self._num_samples * level)

        # take +- delta of events
        delta_index = int(self._num_samples * delta)

        index_min = max(0, index_at_level - delta_index)
        index_max = min(self._num_samples,
                        index_at_level + delta_index)

        if index_max - index_min <= 10:
            raise ValueError('Number of samples is too low!')

        return index_min, index_max

    def contour(self, level=0.5, delta=0.001):
        """Get zenith/azimuth paris of points that lie with the specified
        contour [level - delta, level + delta].

        Parameters
        ----------
        level : float, optional
            The contour level. Example: a level of 0.7 means that 70% of events
            are within this contour.
        delta : float, optional
            The contour is provided by selecting directions from the sampled
            ones which have cdf values within [level - delta, level + delta].
            The smaller delta, the more accurate the contour will be. However,
            the number of available sample points for the contour will also
            decrease.

        Returns
        -------
        np.array, np.array
            The zenith/azimuth pairs that lie within the contour
            [level - delta, level + delta].
        """
        index_min, index_max = self._get_level_indices(level, delta)
        return (self.zenith_s[index_min:index_max],
                self.azimuth_s[index_min:index_max])

    def contour_dir(self, level=0.5, delta=0.001):
        """Get direction vectors of points that lie with the specified
        contour [level - delta, level + delta].

        Parameters
        ----------
        level : float, optional
            The contour level. Example: a level of 0.7 means that 70% of events
            are within this contour.
        delta : float, optional
            The contour is provided by selecting directions from the sampled
            ones which have cdf values within [level - delta, level + delta].
            The smaller delta, the more accurate the contour will be. However,
            the number of available sample points for the contour will also
            decrease.

        Returns
        -------
        np.array, np.array, np.array
            The direction vectors that lie within the contour
            [level - delta, level + delta].
        """
        index_min, index_max = self._get_level_indices(level, delta)
        return (self.dir_x_s[index_min:index_max],
                self.dir_y_s[index_min:index_max],
                self.dir_z_s[index_min:index_max])

    def check_coverage(self, dir_x, dir_y, dir_z,
                       quantiles=np.linspace(0.001, 1., 1000)):
        """Check coverage by calculating contaiment for given direction
        vectors.

        Parameters
        ----------
        dir_x : np.array
            The direction vector x component.
        dir_y : np.array
            The direction vector y component.
        dir_z : np.array
            The direction vector z component.
        quantiles : array_like, optional
            The quantiles at which to calculate the containment.

        Returns
        -------
        array_like, np.array
            The quantiles and the containment at each quantile.
        """
        num_events = float(len(dir_x))
        cdf = self.cdf_dir(dir_x, dir_y, dir_z)
        containment = np.empty(len(quantiles))
        for i, q in enumerate(quantiles):

            # q is the central quantile, e.g. q * 100% of all events land
            # in this central region.
            # Find number of events that are within central quantile q:
            containment[i] = np.sum(cdf <= q) / num_events

        return quantiles, containment
