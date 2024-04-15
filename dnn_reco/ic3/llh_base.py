from __future__ import division, print_function
import numpy as np
import healpy as hp
from uncertainties import unumpy, umath
from scipy.stats import multivariate_normal
from scipy.optimize import minimize


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

    def __init__(
        self,
        dir_x,
        dir_y,
        dir_z,
        unc_x,
        unc_y,
        unc_z,
        random_seed=42,
        weighted_normalization=True,
    ):
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
        """
        self.weighted_normalization = weighted_normalization
        self.unc_x = unc_x
        self.unc_y = unc_y
        self.unc_z = unc_z
        self.dir_x, self.dir_y, self.dir_z = self.normalize_dir(
            dir_x, dir_y, dir_z
        )
        self.zenith, self.azimuth = self.get_zenith_azimuth(
            self.dir_x, self.dir_y, self.dir_z
        )

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

    def u_get_zenith_azimuth(self, u_dir_x, u_dir_y, u_dir_z, with_flip=True):
        """Get zeniths and azimuths [in radians] from direction vector.

        Parameters
        ----------
        u_dir_x : unumpy.array
            The x-component of the direction vector with uncertainty.
        u_dir_y : unumpy.array
            The y-component of the direction vector with uncertainty.
        u_dir_z : unumpy.array
            The z-component of the direction vector with uncertainty.
        with_flip : bool, optional
            If True, then the direction vectors show in the opposite direction
            than the zenith/azimuth pairs. This is common within IceCube
            software, since the direction vector shows along the particle
            direction, whereas the zenith/azimuth shows to the source position.

        Returns
        -------
        unumpy.array, unumpy.array
            The zenith and azimuth angles with uncertainties.
        """
        # normalize
        if not self.is_normalized(
            u_dir_x.nominal_value, u_dir_y.nominal_value, u_dir_z.nominal_value
        ):
            u_dir_x, u_dir_y, u_dir_z = self.u_normalize_dir(
                u_dir_x, u_dir_y, u_dir_z
            )

        if with_flip:
            u_dir_x *= -1
            u_dir_y *= -1
            u_dir_z *= -1

        if np.abs(u_dir_z.nominal_value) > 1.0:
            raise ValueError(
                "Z-component |{!r}| > 1!".format(u_dir_z.nominal_value)
            )
        zenith = umath.acos(u_dir_z)
        azimuth = (umath.atan2(u_dir_y, u_dir_x) + 2 * np.pi) % (2 * np.pi)

        return zenith, azimuth

    def get_zenith_azimuth(self, dir_x, dir_y, dir_z, with_flip=True):
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
        if not self.is_normalized(dir_x, dir_y, dir_z):
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
        return -2 * np.log(sigma) - ((x - mu) / sigma) ** 2

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

    def is_normalized(self, dir_x, dir_y, dir_z):
        """Checks if a direction vector is normalized.

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
        bool
            True, if the direction vector is normalized
        """
        norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
        return np.allclose(norm, 1.0)

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
        if self.weighted_normalization:
            if not self.is_normalized(
                u_dir_x.nominal_value,
                u_dir_y.nominal_value,
                u_dir_z.nominal_value,
            ):
                raise NotImplementedError(
                    "Direction vector must be normalized!"
                )
            return u_dir_x, u_dir_y, u_dir_z
        else:
            norm = unumpy.sqrt(u_dir_x**2 + u_dir_y**2 + u_dir_z**2)
            return u_dir_x / norm, u_dir_y / norm, u_dir_z / norm

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
                norm_list = [np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)]
                dir_x = [dir_x]
                dir_y = [dir_y]
                dir_z = [dir_z]
            else:
                was_float = False
                norm_list = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)

            # define constraint
            cons = {"type": "eq", "fun": lambda x: np.linalg.norm(x) - 1}

            # define cost function
            def cost(dir_normed, dx, dy, dz):
                x, y, z = dir_normed
                c = ((x - dx) / self.unc_x) ** 2
                c += ((y - dy) / self.unc_y) ** 2
                c += ((z - dz) / self.unc_z) ** 2
                return c

            def minimize_vector(dx, dy, dz, norm):
                x0 = [dx / norm, dy / norm, dz / norm]
                result = minimize(
                    cost,
                    x0,
                    args=(dx, dy, dz),
                    constraints=cons,
                    options={"ftol": 1e-04},
                )
                return result.x

            size = len(dir_x)
            dir_n = np.empty((size, 3))
            for i, (dx, dy, dz, norm) in enumerate(
                zip(dir_x, dir_y, dir_z, norm_list)
            ):
                dir_n[i] = minimize_vector(dx, dy, dz, norm)

            if was_float:
                return dir_n[0, 0], dir_n[0, 1], dir_n[0, 2]
            else:
                return dir_n[:, 0], dir_n[:, 1], dir_n[:, 2]

        # Naive scaling: create unit vector by dividing by norm
        else:
            norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
            return dir_x / norm, dir_y / norm, dir_z / norm

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
            The cumulative probability for the given zenith/azimuth pairs.
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
            The cumulative probability for the given direction vectors.
        """
        raise NotImplementedError

    def _get_level_indices(self, level=0.5, delta=0.01):
        """Get indices of sampled directions, which belong to the specified
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
        return (
            self.zenith_s[index_min:index_max],
            self.azimuth_s[index_min:index_max],
        )

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
        return (
            self.dir_x_s[index_min:index_max],
            self.dir_y_s[index_min:index_max],
            self.dir_z_s[index_min:index_max],
        )

    def check_coverage(
        self, dir_x, dir_y, dir_z, quantiles=np.linspace(0.001, 1.0, 1000)
    ):
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

    def contour_area(self, levels, nside):
        """Get the area inside a contour of a given level in square degrees.

        Parameters
        ----------
        levels : float or array_like
            The level or levels for which to compute the contained area.
            Must be a value in (0, 1).
        nside : int
            The nside parameter of the HEALpix.
            The higher nside, the more accurate, but also slower to compute.

        Returns
        -------
        array_like
            The contained area in square degrees for each specified level
        """
        if isinstance(levels, float):
            levels = [levels]

        # area of one HEALPix with given nside in square degrees
        pix_area = hp.nside2pixarea(nside, degrees=True)
        npix = hp.nside2npix(nside)
        cdf_values = self.cdf_dir(*hp.pix2vec(nside, range(npix)))

        # number of HEALPix pixels within contour
        contour_areas = []
        for level in levels:
            num_pixels_inside = np.sum(cdf_values <= level)
            contour_areas.append(num_pixels_inside * pix_area)
        return contour_areas


class DNN_LLH_Base_Elliptical(DNN_LLH_Base):
    """The DNN LLH base class for calculating elliptical PDFs obtained from
    the DNN reco.

    Attributes
    ----------
    azimuth : float
        The best fit azimuth. This is the output of the DNN reco.
    zenith : float
        The best fit zenith. This is the output of the DNN reco.
    cov : array_like
        The covariance matrix for zenith and azimuth.
        This is obtained from the uncertainty estimate of the DNN reco.
        Shape: [2, 2]
    dir_x : float
        The best fit direction vector x component.
        This is the output of the DNN reco for the x-component.
    dir_y : float
        The best fit direction vector y component.
        This is the output of the DNN reco for the y-component.
    dir_z : float
        The best fit direction vector z component.
        This is the output of the DNN reco for the z-component.
    """

    def __init__(
        self,
        zenith,
        azimuth,
        cov,
        num_samples=1000000,
        random_seed=42,
        weighted_normalization=True,
        fix_delta=True,
    ):
        """Initialize DNN LLH object.

        Parameters
        ----------
        zenith : float
            The best fit zenith. This is the output of the DNN reco.
        azimuth : float
            The best fit azimuth. This is the output of the DNN reco.
        cov : array_like
            The covariance matrix for zenith and azimuth.
            This is obtained from the uncertainty estimate of the DNN reco.
            Shape: [2, 2]
        num_samples : int, optional
            Number of samples to sample for internal calculations.
            The more samples, the more accurate, but also slower.
        random_seed : int, optional
            Random seed for sampling.
        weighted_normalization : bool, optional
            If True the normalization vectors get normalized according to the
            uncertainty on each of its components.
            If False, the vectors get scaled by their norm to obtain unit
            vectors.
        fix_delta : bool, optional
            If True, the sampled direction vectors will sampled in a way such
            that the deltas of the angles: abs(azimuth - sampled_azimuth) and
            abs(zenith - sampled_zenith) follow the expected distribution.
        """
        self._fix_delta = fix_delta
        self._num_samples = num_samples
        self.weighted_normalization = weighted_normalization
        self.zenith = zenith
        self.azimuth = azimuth

        self.dir_x, self.dir_y, self.dir_z = self.get_dir_vec(
            self.zenith, self.azimuth
        )

        self._random_seed = random_seed
        self._random_state = np.random.RandomState(self._random_seed)

        self.cov = cov

        # sample contours
        self.zenith_s, self.azimuth_s = self.sample(self._num_samples)

        self.neg_llh_values = -self.log_prob(self.zenith_s, self.azimuth_s)

        # sort sampled points according to neg llh
        sorted_indices = np.argsort(self.neg_llh_values)
        self.zenith_s = self.zenith_s[sorted_indices]
        self.azimuth_s = self.azimuth_s[sorted_indices]
        self.neg_llh_values = self.neg_llh_values[sorted_indices]

        # get sampled coordinate vectors
        self.dir_x_s, self.dir_y_s, self.dir_z_s = self.get_dir_vec(
            self.zenith_s, self.azimuth_s
        )

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
        return multivariate_normal.logpdf(
            np.array([zenith, azimuth]).T,
            mean=np.array([self.zenith, self.azimuth]).T,
            cov=self.cov,
        )

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
        return self.log_prob(*self.get_zenith_azimuth(dir_x, dir_y, dir_z))

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
        if self._fix_delta:
            deltas = self._random_state.multivariate_normal(
                [0.0, self.azimuth], cov=self.cov, size=n
            )
            deltas_zenith, azimuth = deltas[:, 0], deltas[:, 1]

            def fix_delta(delta, d, min_value, max_value):

                # check if over bound
                mask_over_bound = np.logical_or(
                    d + delta > max_value, d + delta < min_value
                )

                # check if going in other direction is in bounds
                mask_allowed = np.logical_and(
                    d - delta < max_value, d - delta > min_value
                )
                mask_fixable = np.logical_and(mask_over_bound, mask_allowed)
                mask_on_boundary = np.logical_and(
                    mask_over_bound, ~mask_allowed
                )

                # For those events that are over bounds in either direction,
                # choose the furthest boundary
                dist_to_max = max_value - d
                dist_to_min = d - min_value
                mask_greater = dist_to_max > dist_to_min

                mask_on_right_boundary = np.logical_and(
                    mask_on_boundary, mask_greater
                )
                mask_on_left_boundary = np.logical_and(
                    mask_on_boundary, ~mask_greater
                )

                # Fix deltas
                delta[mask_on_left_boundary] = -dist_to_min
                delta[mask_on_right_boundary] = +dist_to_max
                delta[mask_fixable] *= -1.0
                return delta

            deltas_zenith = fix_delta(deltas_zenith, self.zenith, 0.0, np.pi)

            zenith = self.zenith + deltas_zenith
            azimuth = azimuth % (2 * np.pi)
        else:
            res = self._random_state.multivariate_normal(
                [self.zenith, self.azimuth], cov=self.cov, size=n
            )
            zenith, azimuth = res[:, 0], res[:, 1]

        return zenith, azimuth

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
        return self.get_dir_vec(*self.sample(n))

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
            The cumulative probability for the given zenith/azimuth pairs.
        """
        neg_llh = -self.log_prob(zenith, azimuth)
        pos = np.searchsorted(self.neg_llh_values, neg_llh)
        cdf = 1.0 * pos / self._num_samples
        return cdf

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
            The cumulative probability for the given direction vectors.
        """
        return self.cdf(*self.get_zenith_azimuth(dir_x, dir_y, dir_z))

    def _get_level_indices(self, level=0.5, delta=0.01):
        """Get indices of sampled directions, which belong to the specified
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
        assert level >= 0.0, level
        assert level <= 1.0, level

        index_at_level = int(self._num_samples * level)

        # take +- delta of events
        delta_index = int(self._num_samples * delta)

        index_min = max(0, index_at_level - delta_index)
        index_max = min(self._num_samples, index_at_level + delta_index)

        if index_max - index_min <= 10:
            raise ValueError("Number of samples is too low!")

        return index_min, index_max
