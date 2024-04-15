from __future__ import division, print_function
import numpy as np
import healpy as hp
import uncertainties
from uncertainties import unumpy, ufloat, covariance_matrix
from scipy.stats import multivariate_normal

from dnn_reco.ic3.llh_base import DNN_LLH_Base, DNN_LLH_Base_Elliptical


class DNN_LLH_Circular_Dir(DNN_LLH_Base_Elliptical):
    """The DNN LLH class for calculating circular PDFs obtained from
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
        dir_x,
        dir_y,
        dir_z,
        unc_x,
        unc_y,
        unc_z,
        num_samples=1000000,
        random_seed=42,
        weighted_normalization=True,
        fix_delta=True,
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
        self.weighted_normalization = weighted_normalization
        u_dir_x = ufloat(dir_x, unc_x)
        u_dir_y = ufloat(dir_y, unc_y)
        u_dir_z = ufloat(dir_z, unc_z)
        u_dir_x, u_dir_y, u_dir_z = self.u_normalize_dir(
            u_dir_x, u_dir_y, u_dir_z
        )

        # Assign values with propagated and normalized vector
        u_zenith, u_azimuth = self.u_get_zenith_azimuth(
            u_dir_x, u_dir_y, u_dir_z
        )
        self.dir_x = u_dir_x.nominal_value
        self.dir_y = u_dir_y.nominal_value
        self.dir_z = u_dir_z.nominal_value

        unc_zenith = unumpy.std_devs(u_zenith)
        unc_azimuth = unumpy.std_devs(u_azimuth)
        zenith = unumpy.nominal_values(u_zenith)
        azimuth = unumpy.nominal_values(u_azimuth)

        # calculate circular error radius
        # (Note we want to get 'average' circular error, therefore divide by 2)
        circular_var = (
            unc_zenith**2 + unc_azimuth**2 * np.sin(zenith) ** 2
        ) / 2.0
        cov = np.diag([circular_var, circular_var])
        DNN_LLH_Base_Elliptical.__init__(
            self,
            zenith,
            azimuth,
            cov,
            num_samples,
            random_seed,
            weighted_normalization,
            fix_delta,
        )


class DNN_LLH_Circular(DNN_LLH_Base_Elliptical):
    """The DNN LLH class for calculating circular PDFs obtained from
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
        unc_zenith,
        unc_azimuth,
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
        unc_zenith : float
            The estimated uncertainty on the zenith angle.
            This is the output of the DNN reco.
        unc_azimuth : float
            The estimated uncertainty on the azimuth angle.
            This is the output of the DNN reco.
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
        # calculate circular error radius
        # (Note we want to get 'average' circular error, therefore divide by 2)
        circular_var = (
            unc_zenith**2 + unc_azimuth**2 * np.sin(zenith) ** 2
        ) / 2.0
        cov = np.diag([circular_var, circular_var])
        DNN_LLH_Base_Elliptical.__init__(
            self,
            zenith,
            azimuth,
            cov,
            num_samples,
            random_seed,
            weighted_normalization,
            fix_delta,
        )


class DNN_LLH_Elliptical_Dir(DNN_LLH_Base_Elliptical):
    """The DNN LLH class for calculating elliptical PDFs obtained from
    the DNN reco.

    Attributes
    ----------
    azimuth : float
        The best fit azimuth. This is the output of the DNN reco.
    zenith : float
        The best fit zenith. This is the output of the DNN reco.
    unc_zenith : float
        The estimated uncertainty on the zenith angle.
        This is the output of the DNN reco.
    unc_azimuth : float
        The estimated uncertainty on the azimuth angle.
        This is the output of the DNN reco.
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
        dir_x,
        dir_y,
        dir_z,
        unc_x,
        unc_y,
        unc_z,
        num_samples=1000000,
        random_seed=42,
        weighted_normalization=True,
        fix_delta=True,
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
        self.weighted_normalization = weighted_normalization
        u_dir_x = ufloat(dir_x, unc_x)
        u_dir_y = ufloat(dir_y, unc_y)
        u_dir_z = ufloat(dir_z, unc_z)
        u_dir_x, u_dir_y, u_dir_z = self.u_normalize_dir(
            u_dir_x, u_dir_y, u_dir_z
        )

        # Assign values with propagated and normalized vector
        u_zenith, u_azimuth = self.u_get_zenith_azimuth(
            u_dir_x, u_dir_y, u_dir_z
        )
        self.dir_x = u_dir_x.nominal_value
        self.dir_y = u_dir_y.nominal_value
        self.dir_z = u_dir_z.nominal_value
        self.zenith = unumpy.nominal_values(u_zenith)
        self.azimuth = unumpy.nominal_values(u_azimuth)
        self.unc_zenith = unumpy.std_devs(u_zenith)
        self.unc_azimuth = unumpy.std_devs(u_azimuth)
        cov = np.array(covariance_matrix([u_zenith, u_azimuth]))
        DNN_LLH_Base_Elliptical.__init__(
            self,
            self.zenith,
            self.azimuth,
            cov,
            num_samples,
            random_seed,
            weighted_normalization,
            fix_delta,
        )


class DNN_LLH_Elliptical(DNN_LLH_Base_Elliptical):
    """The DNN LLH class for calculating elliptical PDFs obtained from
    the DNN reco.

    Attributes
    ----------
    azimuth : float
        The best fit azimuth. This is the output of the DNN reco.
    zenith : float
        The best fit zenith. This is the output of the DNN reco.
    unc_zenith : float
        The estimated uncertainty on the zenith angle.
        This is the output of the DNN reco.
    unc_azimuth : float
        The estimated uncertainty on the azimuth angle.
        This is the output of the DNN reco.
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
        unc_zenith,
        unc_azimuth,
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
        unc_zenith : float
            The estimated uncertainty on the zenith angle.
            This is the output of the DNN reco.
        unc_azimuth : float
            The estimated uncertainty on the azimuth angle.
            This is the output of the DNN reco.
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
        cov = np.diag([unc_zenith**2, unc_azimuth**2])
        DNN_LLH_Base_Elliptical.__init__(
            self,
            zenith,
            azimuth,
            cov,
            num_samples,
            random_seed,
            weighted_normalization,
            fix_delta,
        )


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

    def __init__(
        self,
        dir_x,
        dir_y,
        dir_z,
        unc_x,
        unc_y,
        unc_z,
        nside=256,
        random_seed=42,
        scale_unc=True,
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
        nside : int, optional
            Description
        random_seed : int, optional
            Random seed for sampling.
        scale_unc : bool, optional
            Due to the normalization of the direction vectors, the components
            of the vector are correlated, hence the actual spread in sampled
            direction vectors shrinks. The nn model predicts the Gaussian
            Likelihood of the normalized vectors (if normalization is included)
            in network model. In this case, the uncertainties of the
            direction vector components can be scaled to account for this
            correlation.
            If set to True, the uncertainties will be scaled.
        weighted_normalization : bool, optional
            If True the normalization vectors get normalized according to the
            uncertainty on each of its components.
            If False, the vectors get scaled by their norm to obtain unit
            vectors.
        """

        # call init from base class
        DNN_LLH_Base.__init__(
            self,
            dir_x,
            dir_y,
            dir_z,
            unc_x,
            unc_y,
            unc_z,
            random_seed,
            weighted_normalization,
        )

        def _setup(nside):

            self.cov = np.diag([self.unc_x**2, self.unc_y**2, self.unc_z**2])

            # compute pdf for each pixel
            self.nside = nside
            self._n_order = self._nside2norder()
            self.npix = hp.nside2npix(nside)
            self.dir_x_s, self.dir_y_s, self.dir_z_s = hp.pix2vec(
                nside, range(self.npix)
            )

            self.neg_llh_values = -self.log_prob_dir(
                self.dir_x_s, self.dir_y_s, self.dir_z_s
            )

            # sort directions according to neg llh
            sorted_indices = np.argsort(self.neg_llh_values)
            self.dir_x_s = self.dir_x_s[sorted_indices]
            self.dir_y_s = self.dir_y_s[sorted_indices]
            self.dir_z_s = self.dir_z_s[sorted_indices]
            self.neg_llh_values = self.neg_llh_values[sorted_indices]
            self.ipix_list = sorted_indices

            # get zenith and azimuth
            self.zenith_s, self.azimuth_s = self.get_zenith_azimuth(
                self.dir_x_s, self.dir_y_s, self.dir_z_s
            )

            # get normalized probabilities and cdf
            prob = np.exp(-self.neg_llh_values)
            self.prob_values = prob / np.sum(prob)
            self.cdf_values = np.cumsum(self.prob_values)

        # -------------------------
        # scale up unc if necessary
        # -------------------------
        self.scale_unc = scale_unc

        def _scale(nside):
            # set up once to be able to perform scaling
            _setup(nside=nside)
            dir_x_s, dir_y_s, dir_z_s = self.sample_dir(10000)
            # print('scaling x by:', self.unc_x / np.std(dir_x_s))
            # print('scaling y by:', self.unc_y / np.std(dir_y_s))
            # print('scaling z by:', self.unc_z / np.std(dir_z_s))
            self.unc_x *= self.unc_x / np.std(dir_x_s)
            self.unc_y *= self.unc_y / np.std(dir_y_s)
            self.unc_z *= self.unc_z / np.std(dir_z_s)

        if self.scale_unc:
            _scale(nside=nside)

        # -------------------------
        _setup(nside=nside)

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
        from scipy.stats import multivariate_normal

        return multivariate_normal.logpdf(
            np.array([dir_x, dir_y, dir_z]).T,
            mean=np.array([self.dir_x, self.dir_y, self.dir_z]).T,
            cov=self.cov,
        )

        # log_p = self.log_gauss(dir_x, self.dir_x, self.unc_x)
        # log_p += self.log_gauss(dir_y, self.dir_y, self.unc_y)
        # log_p += self.log_gauss(dir_z, self.dir_z, self.unc_z)
        # return log_p

    def _nside2norder(self):
        """
        Give the HEALpix order for the given HEALpix nside parameter.

        Credit goes to:
            https://git.rwth-aachen.de/astro/astrotools/blob/master/
            astrotools/healpytools.py

        Returns
        -------
        int
            norder of the healpy pixelization

        Raises
        ------
        ValueError
            If nside is not 2**norder.
        """
        norder = np.log2(self.nside)
        if not (norder.is_integer()):
            raise ValueError("Wrong nside number (it is not 2**norder)")
        return int(norder)

    def _sample_from_ipix(self, ipix, nest=False):
        """
        Sample vectors from a uniform distribution within a HEALpixel.

        Credit goes to
        https://git.rwth-aachen.de/astro/astrotools/blob/master/
        astrotools/healpytools.py

        :param ipix: pixel number(s)
        :param nest: set True in case you work with healpy's nested scheme
        :return: vectors containing events from the pixel(s) specified in ipix

        Parameters
        ----------
        ipix : int, list of int
            Healpy pixels.
        nest : bool, optional
            Set to True in case healpy's nested scheme is used.

        Returns
        -------
        np.array, np.array, np.array
            The sampled direction vector components.
        """
        if not nest:
            ipix = hp.ring2nest(self.nside, ipix=ipix)

        n_up = 29 - self._n_order
        i_up = ipix * 4**n_up
        i_up += self._random_state.randint(0, 4**n_up, size=np.size(ipix))
        return hp.pix2vec(nside=2**29, ipix=i_up, nest=True)

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
        # sample random healpy pixels given their probability
        indices = np.searchsorted(self.cdf_values, self._random_state.rand(n))
        indices[indices > self.npix - 1] = self.npix - 1

        # get the healpy pixels
        ipix = self.ipix_list[indices]

        # sample directions within these pixels
        dir_x, dir_y, dir_z = self._sample_from_ipix(ipix)

        return dir_x, dir_y, dir_z

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
        if not self.is_normalized(dir_x, dir_y, dir_z):
            print("cdf_dir is normalizing direction vectors")
            dir_x, dir_y, dir_z = self.normalize_dir(dir_x, dir_y, dir_z)

        neg_llh = -self.log_prob_dir(dir_x, dir_y, dir_z)
        pos = np.searchsorted(self.neg_llh_values, neg_llh)
        pos_clipped = np.clip(pos, 0, self.npix - 1)
        assert np.abs(pos - pos_clipped).all() <= 1
        cdf = self.cdf_values[pos_clipped]
        return cdf

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

        index_min = np.searchsorted(self.cdf_values, level - delta)
        index_max = min(
            self.npix, np.searchsorted(self.cdf_values, level + delta)
        )

        if index_max - index_min <= 10:
            raise ValueError("Number of samples is too low!")

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

    def __init__(
        self,
        dir_x,
        dir_y,
        dir_z,
        unc_x,
        unc_y,
        unc_z,
        propagate_errors=False,
        num_samples=1000000,
        random_seed=42,
        scale_unc=True,
        weighted_normalization=True,
        fix_delta=True,
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
        propagate_errors : bool, optional
            Propagate errors and account for correlations.
        num_samples : int, optional
            Number of samples to sample for internal calculations.
            The more samples, the more accurate, but also slower.
        random_seed : int, optional
            Random seed for sampling.
        scale_unc : bool, optional
            Due to the normalization of the direction vectors, the components
            of the vector are correlated, hence the actual spread in sampled
            direction vectors shrinks. The nn model predicts the Gaussian
            Likelihood of the normalized vectors (if normalization is included)
            in network model. In this case, the uncertainties of the
            direction vector components can be scaled to account for this
            correlation.
            If set to True, the uncertainties will be scaled.
        weighted_normalization : bool, optional
            If True the normalization vectors get normalized according to the
            uncertainty on each of its components.
            If False, the vectors get scaled by their norm to obtain unit
            vectors.
        fix_delta : bool, optional
            If True, the sampled direction vectors will sampled in a way such
            that the deltas: abs(dir_i - sampled_dir_i) follows the expected
            distribution.
        """

        # call init from base class
        DNN_LLH_Base.__init__(
            self,
            dir_x,
            dir_y,
            dir_z,
            unc_x,
            unc_y,
            unc_z,
            random_seed,
            weighted_normalization,
        )

        self._num_samples = num_samples
        self.propagate_errors = propagate_errors
        self._fix_delta = fix_delta
        if self.propagate_errors:
            # propagate errors
            u_dir_x = unumpy.uarray(dir_x, unc_x)
            u_dir_y = unumpy.uarray(dir_y, unc_y)
            u_dir_z = unumpy.uarray(dir_z, unc_z)
            u_dir_x, u_dir_y, u_dir_z = self.u_normalize_dir(
                u_dir_x, u_dir_y, u_dir_z
            )

            # Assign values with propagated and normalized vector
            self.unc_x = u_dir_x.std_dev
            self.unc_y = u_dir_y.std_dev
            self.unc_z = u_dir_z.std_dev
            self.dir_x = u_dir_x.nominal_value
            self.dir_y = u_dir_y.nominal_value
            self.dir_z = u_dir_z.nominal_value
            self.cov_matrix = np.array(
                uncertainties.covariance_matrix([u_dir_x, u_dir_y, u_dir_z])
            )
            self.dist = multivariate_normal(
                mean=(self.dir_x, self.dir_y, self.dir_z),
                cov=self.cov_matrix,
                allow_singular=True,
            )
        else:
            self.unc_x = unc_x
            self.unc_y = unc_y
            self.unc_z = unc_z
            self.dir_x, self.dir_y, self.dir_z = self.normalize_dir(
                dir_x, dir_y, dir_z
            )

        # -------------------------
        # scale up unc if necessary
        # -------------------------
        self.scale_unc = scale_unc

        if self.scale_unc:

            def _scale():
                dir_x_s, dir_y_s, dir_z_s = self.sample_dir(
                    min(self._num_samples, 1000)
                )
                # print('scaling x by:', self.unc_x / np.std(dir_x_s))
                # print('scaling y by:', self.unc_y / np.std(dir_y_s))
                # print('scaling z by:', self.unc_z / np.std(dir_z_s))
                self.unc_x *= self.unc_x / np.std(dir_x_s)
                self.unc_y *= self.unc_y / np.std(dir_y_s)
                self.unc_z *= self.unc_z / np.std(dir_z_s)

            _scale()

        # -------------------------

        self.zenith, self.azimuth = self.get_zenith_azimuth(
            self.dir_x, self.dir_y, self.dir_z
        )
        # sample contours
        self.dir_x_s, self.dir_y_s, self.dir_z_s = self.sample_dir(
            self._num_samples
        )

        self.neg_llh_values = -self.log_prob_dir(
            self.dir_x_s, self.dir_y_s, self.dir_z_s
        )

        # sort sampled points according to neg llh
        sorted_indices = np.argsort(self.neg_llh_values)
        self.dir_x_s = self.dir_x_s[sorted_indices]
        self.dir_y_s = self.dir_y_s[sorted_indices]
        self.dir_z_s = self.dir_z_s[sorted_indices]
        self.neg_llh_values = self.neg_llh_values[sorted_indices]

        # get sampled zenith and azimuth
        self.zenith_s, self.azimuth_s = self.get_zenith_azimuth(
            self.dir_x_s, self.dir_y_s, self.dir_z_s
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
        if self._fix_delta:
            delta_x = self._random_state.normal(0.0, self.unc_x, n)
            delta_y = self._random_state.normal(0.0, self.unc_y, n)
            delta_z = self._random_state.normal(0.0, self.unc_z, n)

            def fix_delta(delta, d):
                # return delta
                mask_over_bound = np.abs(d + delta) > 1.0

                # see if these can be fixed by going in the other direction
                mask_allowed = np.abs(d - delta) < 1.0
                mask_fixable = np.logical_and(mask_over_bound, mask_allowed)
                mask_on_boundary = np.logical_and(
                    mask_over_bound, ~mask_allowed
                )

                # For those events that are over bounds in either direction,
                # choose the furthest boundary
                delta_max = 1 + np.abs(d)
                mask_on_left_boundary = np.logical_and(
                    mask_on_boundary, d > 0.0
                )
                mask_on_right_boundary = np.logical_and(
                    mask_on_boundary, d < 0.0
                )

                delta[mask_on_left_boundary] = -delta_max
                delta[mask_on_right_boundary] = +delta_max

                # fix directions which are fixable
                delta[mask_fixable] *= -1.0
                return delta

            dir_x_s = self.dir_x + fix_delta(delta_x, self.dir_x)
            dir_y_s = self.dir_y + fix_delta(delta_y, self.dir_y)
            dir_z_s = self.dir_z + fix_delta(delta_z, self.dir_z)

        else:
            dir_x_s = self._random_state.normal(self.dir_x, self.unc_x, n)
            dir_y_s = self._random_state.normal(self.dir_y, self.unc_y, n)
            dir_z_s = self._random_state.normal(self.dir_z, self.unc_z, n)
        dir_x_s, dir_y_s, dir_z_s = self.normalize_dir(
            dir_x_s, dir_y_s, dir_z_s
        )
        return dir_x_s, dir_y_s, dir_z_s

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
        if not self.is_normalized(dir_x, dir_y, dir_z):
            print("cdf_dir is normalizing direction vectors")
            dir_x, dir_y, dir_z = self.normalize_dir(dir_x, dir_y, dir_z)

        neg_llh = -self.log_prob_dir(dir_x, dir_y, dir_z)
        pos = np.searchsorted(self.neg_llh_values, neg_llh)
        cdf = 1.0 * pos / self._num_samples
        return cdf

    def _get_level_indices(self, level=0.5, delta=0.001):
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
