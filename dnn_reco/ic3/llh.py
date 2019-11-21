#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import healpy as hp
import uncertainties
from uncertainties import unumpy
from scipy.stats import multivariate_normal

from dnn_reco.ic3.llh_base import DNN_LLH_Base


class DNN_LLH_normalized(DNN_LLH_Base):

    """A class for calculating the PDF obtained from the DNN reco for models
    that estimate the direction vector components and their uncertainty in
    independent 1D Gaussian Likelihoods, while the direction vector is
    normalized within the neural network model.

    Note: if the used neural network model does not fulfill these requirements,
    the PDF will be incorrect!

    Attributes
    ----------
    cdf_values : np.array
        The cumulative probability values for the sorted internal directions.
    prob_values : np.array
        The normalizes probabilities for the sorted internal directions.
    dir_x_s : np.array
        The x component of the sorted internal direction vectors.
    dir_y_s : np.array
        The y component of the sorted internal direction vectors.
    dir_z_s : np.array
        The z component of the sorted internal direction vectors.
    neg_llh_values : np.array
        The negative log probability values for the sorted internal directions.
    npix : int
        Number of healpy pixels.
    nside : int
        The nside parameter for the healpy pixels. This defines the resolution
        and accuracy of the PDF. The higher nside, the better the resolution,
        but also the slower it is.
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
    """

    def __init__(self, dir_x, dir_y, dir_z, unc_x, unc_y, unc_z,
                 nside=256, random_seed=42):
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
        nside : int, optional
            Description
        random_seed : int, optional
            Random seed for sampling.
        """

        # call init from base class
        DNN_LLH_Base.__init__(self, dir_x, dir_y, dir_z, unc_x, unc_y, unc_z,
                              random_seed)

        # compute pdf for each pixel
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.dir_x_s, self.dir_y_s, self.dir_z_s = \
            hp.pix2vec(nside, range(self.npix))

        self.neg_llh_values = -self.log_prob_dir(
            self.dir_x_s, self.dir_y_s, self.dir_z_s)

        # sort directions according to neg llh
        sorted_indices = np.argsort(self.neg_llh_values)
        self.dir_x_s = self.dir_x_s[sorted_indices]
        self.dir_y_s = self.dir_y_s[sorted_indices]
        self.dir_z_s = self.dir_z_s[sorted_indices]
        self.neg_llh_values = self.neg_llh_values[sorted_indices]

        # get zenith and azimuth
        self.zenith_s, self.azimuth_s = self.get_zenith_azimuth(
                        self.dir_x_s, self.dir_y_s, self.dir_z_s)

        # get normalized probabilities and cdf
        prob = np.exp(-self.neg_llh_values)
        self.prob_values = prob / np.sum(prob)
        self.cdf_values = np.cumsum(self.prob_values)

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

    def sample_dir(self, n):
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
        n_sampled = 0.
        dir_x = []
        dir_y = []
        dir_z = []
        while not n_sampled >= n:
            rand = self._random_state.uniform(size=self.npix)
            event_at_ipix = self.prob_values >= rand

            n_sampled += np.sum(event_at_ipix)
            dir_x.append(self.dir_x_s[event_at_ipix])
            dir_y.append(self.dir_y_s[event_at_ipix])
            dir_z.append(self.dir_z_s[event_at_ipix])

        dir_x = np.concatenate(dir_x)
        dir_y = np.concatenate(dir_y)
        dir_z = np.concatenate(dir_z)

        return dir_x[:n], dir_y[:n], dir_z[:n]

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
        pos_clipped = np.clip(pos, 0, self.npix - 1)
        assert np.abs(pos - pos_clipped).all() <= 1
        cdf = self.cdf_values[pos_clipped]
        return cdf

    def _get_level_indices(self, level=0.5, delta=0.01):
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

        index_min = np.searchsorted(self.cdf_values, level - delta)
        index_max = min(self.npix,
                        np.searchsorted(self.cdf_values, level + delta))

        if index_max - index_min <= 10:
            raise ValueError('Number of samples is too low!')

        return index_min, index_max


class DNN_LLH(DNN_LLH_Base):

    """A class for calculating the PDF obtained from the DNN reco for models
    that estimate the direction vector components and their uncertainty in
    independent 1D Gaussian Likelihoods, while the direction vector is
    NOT normalized within the neural network model.

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
                 propagate_errors=False, num_samples=1000000, random_seed=42):
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
        propagate_errors : bool, optional
            Propagate errors and account for correlations.
        num_samples : int, optional
            Number of samples to sample for internal calculations.
            The more samples, the more accurate, but also slower.
        random_seed : int, optional
            Random seed for sampling.
        """

        # call init from base class
        DNN_LLH_Base.__init__(self, dir_x, dir_y, dir_z, unc_x, unc_y, unc_z,
                              random_seed)

        self.propagate_errors = propagate_errors
        if self.propagate_errors:
            # propagate errors
            u_dir_x = unumpy.uarray(dir_x, unc_x)
            u_dir_y = unumpy.uarray(dir_y, unc_y)
            u_dir_z = unumpy.uarray(dir_z, unc_z)
            u_dir_x, u_dir_y, u_dir_z = self.u_normalize_dir(
                                                u_dir_x, u_dir_y, u_dir_z)

            # Assign values with propagated and normalized vector
            self.unc_x = u_dir_x.std_dev
            self.unc_y = u_dir_y.std_dev
            self.unc_z = u_dir_z.std_dev
            self.dir_x = u_dir_x.nominal_value
            self.dir_y = u_dir_y.nominal_value
            self.dir_z = u_dir_z.nominal_value
            self.cov_matrix = np.array(
                uncertainties.covariance_matrix([u_dir_x, u_dir_y, u_dir_z]))
            self.dist = multivariate_normal(
                mean=(self.dir_x, self.dir_y, self.dir_z),
                cov=self.cov_matrix, allow_singular=True)
        else:
            self.unc_x = unc_x
            self.unc_y = unc_y
            self.unc_z = unc_z
            self.dir_x, self.dir_y, self.dir_z = \
                self.normalize_dir(dir_x, dir_y, dir_z)

        self._num_samples = num_samples
        self.zenith, self.azimuth = self.get_zenith_azimuth(
                                        self.dir_x, self.dir_y, self.dir_z)
        # sample contours
        self.dir_x_s, self.dir_y_s, self.dir_z_s = \
            self.sample_dir(self._num_samples)

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

    def sample_dir(self, n):
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

    def u_normalize_dir(self, u_dir_x, u_dir_y, u_dir_z):
        """Normalize direction vector

        Parameters
        ----------
        u_dir_x : unumpy.array
            The x-component of the direction vector with uncertainty.
        u_dir_y : unumpy.array
            The y-component of the direction vector with uncertainty.
        u_dir_z : unumpy.array
            The z-component of the direction vector with uncertainty.

        Returns
        -------
        unumpy.array, unumpy.array, unumpy.array
            The normalized direction vector components with uncertainties.
        """
        norm = unumpy.sqrt(u_dir_x**2 + u_dir_y**2 + u_dir_z**2)
        return u_dir_x/norm, u_dir_y/norm, u_dir_z/norm

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
