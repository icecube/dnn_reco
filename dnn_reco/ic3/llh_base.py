#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import healpy as hp
from uncertainties import unumpy
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from tqdm import tqdm


class DNN_LLH_Base(object):

    """The DNN LLH base class for calculating the PDF obtained from the DNN
    reco.

    Attributes
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
    """

    def __init__(self, dir_x, dir_y, dir_z, unc_x, unc_y, unc_z,
                 random_seed=42, weighted_normalization=True):
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
        random_seed : int, optional
            Random seed for sampling.
        weighted_normalization : bool, optional
            If True the normalization vectors get normalized according to the
            uncertainty on each of its components.
            If False, the vectors get scaled by their norm to obtain unit
            vectors.

        Deleted Parameters
        ------------------
        nside : int, optional
            Description
        """
        self.weighted_normalization = weighted_normalization
        self.unc_x = unc_x
        self.unc_y = unc_y
        self.unc_z = unc_z
        self.dir_x, self.dir_y, self.dir_z = \
            self.normalize_dir(dir_x, dir_y, dir_z)
        self.zenith, self.azimuth = self.get_zenith_azimuth(
                                        self.dir_x, self.dir_y, self.dir_z)

        self._random_seed = random_seed
        self._random_state = np.random.RandomState(self._random_seed)

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

        if with_flip:
            dir_x = -dir_x
            dir_y = -dir_y
            dir_z = -dir_z

        zenith = np.arccos(np.clip(dir_z, -1, 1))
        azimuth = (np.arctan2(dir_y, dir_x) + 2 * np.pi) % (2 * np.pi)

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
        raise NotImplementedError

    def sample(self, n):
        """Sample direction vectors from the distribution

        Parameters
        ----------
        n : int
            Number of samples to generate.

        Returns
        -------
        np.array, np.array
            Zenith and Azimuth angles in radians of the sampled directions.
        """
        return self.get_zenith_azimuth(*self.sample_dir(n))

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
        raise NotImplementedError

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

        # Scale vectors according to uncertainty of components
        if self.weighted_normalization:

            if isinstance(dir_z, float):
                was_float = True
                dir_x = [dir_x]
                dir_y = [dir_y]
                dir_z = [dir_z]
            else:
                was_float = False

            dir_x_n = []
            dir_y_n = []
            dir_z_n = []
            # for dx, dy, dz in tqdm(zip(dir_x, dir_y, dir_z)):
            for dx, dy, dz in zip(dir_x, dir_y, dir_z):

                def cost(dir_normed):
                    x, y, z = dir_normed
                    c = ((x - dx) / self.unc_x)**2
                    c += ((y - dy) / self.unc_y)**2
                    c += ((z - dz) / self.unc_z)**2
                    return c

                cons = ({'type': 'eq', 'fun': lambda x:  np.linalg.norm(x) - 1})

                x0 = [dx, dy, dz]
                result = minimize(cost, x0, constraints=cons)
                dir_x_n.append(result.x[0])
                dir_y_n.append(result.x[1])
                dir_z_n.append(result.x[2])

            if was_float:
                return dir_x_n[0], dir_y_n[0], dir_z_n[0]
            else:
                return np.array(dir_x_n), np.array(dir_y_n), np.array(dir_z_n)

        # Naive scaling: create unit vector by dividing by norm
        else:
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
        raise NotImplementedError

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
        raise NotImplementedError

    def contour(self, level=0.5, delta=0.01):
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

    def contour_dir(self, level=0.5, delta=0.01):
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
