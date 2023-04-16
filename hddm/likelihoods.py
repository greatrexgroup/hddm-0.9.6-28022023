import pymc as pm
import numpy as np
# from scipy import stats
from hddm.simulators import *

from kabuki.utils import stochastic_from_dist

from hddm.model_config import model_config


np.seterr(divide="ignore")

import hddm



###### DG ADDITION ######
from collections import namedtuple
import scipy.special as sc
import scipy.special as special
from numpy import asarray
from scipy.stats import _distn_infrastructure


def _count(a, axis=None):
    """Count the number of non-masked elements of an array.

    This function behaves like `np.ma.count`, but is much faster
    for ndarrays.
    """
    if hasattr(a, 'count'):
        num = a.count(axis=axis)
        if isinstance(num, np.ndarray) and num.ndim == 0:
            # In some cases, the `count` method returns a scalar array (e.g.
            # np.array(3)), but we want a plain integer.
            num = int(num)
    else:
        if axis is None:
            num = a.size
        else:
            num = a.shape[axis]
    return num


class chi2_gen(_distn_infrastructure.rv_continuous):
    r"""A chi-squared continuous random variable.

    For the noncentral chi-square distribution, see `ncx2`.

    %(before_notes)s

    See Also
    --------
    ncx2

    Notes
    -----
    The probability density function for `chi2` is:

    .. math::

        f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                   x^{k/2-1} \exp \left( -x/2 \right)

    for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
    in the implementation).

    `chi2` takes ``df`` as a shape parameter.

    The chi-squared distribution is a special case of the gamma
    distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
    ``scale = 2``.

    %(after_notes)s

    %(example)s

    """
    def _rvs(self, df, size=None, random_state=None):
        return random_state.chisquare(df, size)

    def _pdf(self, x, df):
        # chi2.pdf(x, df) = 1 / (2*gamma(df/2)) * (x/2)**(df/2-1) * exp(-x/2)
        return np.exp(self._logpdf(x, df))

    def _logpdf(self, x, df):
        return sc.xlogy(df/2.-1, x) - x/2. - sc.gammaln(df/2.) - (np.log(2)*df)/2.

    def _cdf(self, x, df):
        return sc.chdtr(df, x)

    def _sf(self, x, df):
        return sc.chdtrc(df, x)

    def _isf(self, p, df):
        return sc.chdtri(df, p)

    def _ppf(self, p, df):
        return 2*sc.gammaincinv(df/2, p)

    def _stats(self, df):
        mu = df
        mu2 = 2*df
        g1 = 2*np.sqrt(2.0/df)
        g2 = 12.0/df
        return mu, mu2, g1, g2
    
chi2 = chi2_gen(a=0.0, name='chi2')

Power_divergenceResult = namedtuple('Power_divergenceResult',
                                    ('statistic', 'pvalue'))

def _m_broadcast_to(a, shape):
    if np.ma.isMaskedArray(a):
        return np.ma.masked_array(np.broadcast_to(a, shape),
                                  mask=np.broadcast_to(a.mask, shape))
    return np.broadcast_to(a, shape, subok=True)

def _broadcast_shapes(shape1, shape2):
    """
    Given two shapes (i.e. tuples of integers), return the shape
    that would result from broadcasting two arrays with the given
    shapes.

    Examples
    --------
    >>> _broadcast_shapes((2, 1), (4, 1, 3))
    (4, 2, 3)
    """
    d = len(shape1) - len(shape2)
    if d <= 0:
        shp1 = (1,)*(-d) + shape1
        shp2 = shape2
    else:
        shp1 = shape1
        shp2 = (1,)*d + shape2
    shape = []
    for n1, n2 in zip(shp1, shp2):
        if n1 == 1:
            n = n2
        elif n2 == 1 or n1 == n2:
            n = n1
        else:
            raise ValueError(f'shapes {shape1} and {shape2} could not be '
                             'broadcast together')
        shape.append(n)
    return tuple(shape)

# Map from names to lambda_ values used in power_divergence().
_power_div_lambda_names = {
    "pearson": 1,
    "log-likelihood": 0,
    "freeman-tukey": -0.5,
    "mod-log-likelihood": -1,
    "neyman": -2,
    "cressie-read": 2/3,
} 

def power_divergence(f_obs, f_exp=None, ddof=0, axis=0, lambda_=None):
    """Cressie-Read power divergence statistic and goodness of fit test.

    This function tests the null hypothesis that the categorical data
    has the given frequencies, using the Cressie-Read power divergence
    statistic.

    Parameters
    ----------
    f_obs : array_like
        Observed frequencies in each category.
    f_exp : array_like, optional
        Expected frequencies in each category.  By default the categories are
        assumed to be equally likely.
    ddof : int, optional
        "Delta degrees of freedom": adjustment to the degrees of freedom
        for the p-value.  The p-value is computed using a chi-squared
        distribution with ``k - 1 - ddof`` degrees of freedom, where `k`
        is the number of observed frequencies.  The default value of `ddof`
        is 0.
    axis : int or None, optional
        The axis of the broadcast result of `f_obs` and `f_exp` along which to
        apply the test.  If axis is None, all values in `f_obs` are treated
        as a single data set.  Default is 0.
    lambda_ : float or str, optional
        The power in the Cressie-Read power divergence statistic.  The default
        is 1.  For convenience, `lambda_` may be assigned one of the following
        strings, in which case the corresponding numerical value is used::

            String              Value   Description
            "pearson"             1     Pearson's chi-squared statistic.
                                        In this case, the function is
                                        equivalent to `stats.chisquare`.
            "log-likelihood"      0     Log-likelihood ratio. Also known as
                                        the G-test [3]_.
            "freeman-tukey"      -1/2   Freeman-Tukey statistic.
            "mod-log-likelihood" -1     Modified log-likelihood ratio.
            "neyman"             -2     Neyman's statistic.
            "cressie-read"        2/3   The power recommended in [5]_.

    Returns
    -------
    statistic : float or ndarray
        The Cressie-Read power divergence test statistic.  The value is
        a float if `axis` is None or if` `f_obs` and `f_exp` are 1-D.
    pvalue : float or ndarray
        The p-value of the test.  The value is a float if `ddof` and the
        return value `stat` are scalars.

    See Also
    --------
    chisquare

    Notes
    -----
    This test is invalid when the observed or expected frequencies in each
    category are too small.  A typical rule is that all of the observed
    and expected frequencies should be at least 5.

    Also, the sum of the observed and expected frequencies must be the same
    for the test to be valid; `power_divergence` raises an error if the sums
    do not agree within a relative tolerance of ``1e-8``.

    When `lambda_` is less than zero, the formula for the statistic involves
    dividing by `f_obs`, so a warning or error may be generated if any value
    in `f_obs` is 0.

    Similarly, a warning or error may be generated if any value in `f_exp` is
    zero when `lambda_` >= 0.

    The default degrees of freedom, k-1, are for the case when no parameters
    of the distribution are estimated. If p parameters are estimated by
    efficient maximum likelihood then the correct degrees of freedom are
    k-1-p. If the parameters are estimated in a different way, then the
    dof can be between k-1-p and k-1. However, it is also possible that
    the asymptotic distribution is not a chisquare, in which case this
    test is not appropriate.

    This function handles masked arrays.  If an element of `f_obs` or `f_exp`
    is masked, then data at that position is ignored, and does not count
    towards the size of the data set.

    .. versionadded:: 0.13.0

    References
    ----------
    .. [1] Lowry, Richard.  "Concepts and Applications of Inferential
           Statistics". Chapter 8.
           https://web.archive.org/web/20171015035606/http://faculty.vassar.edu/lowry/ch8pt1.html
    .. [2] "Chi-squared test", https://en.wikipedia.org/wiki/Chi-squared_test
    .. [3] "G-test", https://en.wikipedia.org/wiki/G-test
    .. [4] Sokal, R. R. and Rohlf, F. J. "Biometry: the principles and
           practice of statistics in biological research", New York: Freeman
           (1981)
    .. [5] Cressie, N. and Read, T. R. C., "Multinomial Goodness-of-Fit
           Tests", J. Royal Stat. Soc. Series B, Vol. 46, No. 3 (1984),
           pp. 440-464.

    Examples
    --------
    (See `chisquare` for more examples.)

    When just `f_obs` is given, it is assumed that the expected frequencies
    are uniform and given by the mean of the observed frequencies.  Here we
    perform a G-test (i.e. use the log-likelihood ratio statistic):

    >>> from scipy.stats import power_divergence
    >>> power_divergence([16, 18, 16, 14, 12, 12], lambda_='log-likelihood')
    (2.006573162632538, 0.84823476779463769)

    The expected frequencies can be given with the `f_exp` argument:

    >>> power_divergence([16, 18, 16, 14, 12, 12],
    ...                  f_exp=[16, 16, 16, 16, 16, 8],
    ...                  lambda_='log-likelihood')
    (3.3281031458963746, 0.6495419288047497)

    When `f_obs` is 2-D, by default the test is applied to each column.

    >>> obs = np.array([[16, 18, 16, 14, 12, 12], [32, 24, 16, 28, 20, 24]]).T
    >>> obs.shape
    (6, 2)
    >>> power_divergence(obs, lambda_="log-likelihood")
    (array([ 2.00657316,  6.77634498]), array([ 0.84823477,  0.23781225]))

    By setting ``axis=None``, the test is applied to all data in the array,
    which is equivalent to applying the test to the flattened array.

    >>> power_divergence(obs, axis=None)
    (23.31034482758621, 0.015975692534127565)
    >>> power_divergence(obs.ravel())
    (23.31034482758621, 0.015975692534127565)

    `ddof` is the change to make to the default degrees of freedom.

    >>> power_divergence([16, 18, 16, 14, 12, 12], ddof=1)
    (2.0, 0.73575888234288467)

    The calculation of the p-values is done by broadcasting the
    test statistic with `ddof`.

    >>> power_divergence([16, 18, 16, 14, 12, 12], ddof=[0,1,2])
    (2.0, array([ 0.84914504,  0.73575888,  0.5724067 ]))

    `f_obs` and `f_exp` are also broadcast.  In the following, `f_obs` has
    shape (6,) and `f_exp` has shape (2, 6), so the result of broadcasting
    `f_obs` and `f_exp` has shape (2, 6).  To compute the desired chi-squared
    statistics, we must use ``axis=1``:

    >>> power_divergence([16, 18, 16, 14, 12, 12],
    ...                  f_exp=[[16, 16, 16, 16, 16, 8],
    ...                         [8, 20, 20, 16, 12, 12]],
    ...                  axis=1)
    (array([ 3.5 ,  9.25]), array([ 0.62338763,  0.09949846]))

    """
    # Convert the input argument `lambda_` to a numerical value.
    if isinstance(lambda_, str):
        if lambda_ not in _power_div_lambda_names:
            names = repr(list(_power_div_lambda_names.keys()))[1:-1]
            raise ValueError("invalid string for lambda_: {0!r}. "
                             "Valid strings are {1}".format(lambda_, names))
        lambda_ = _power_div_lambda_names[lambda_]
    elif lambda_ is None:
        lambda_ = 1

    f_obs = np.asanyarray(f_obs)
    f_obs_float = f_obs.astype(np.float64)

    if f_exp is not None:
        f_exp = np.asanyarray(f_exp)
        bshape = _broadcast_shapes(f_obs_float.shape, f_exp.shape)
        f_obs_float = _m_broadcast_to(f_obs_float, bshape)
        f_exp = _m_broadcast_to(f_exp, bshape)
        rtol = 1e-4  # to pass existing tests
        with np.errstate(invalid='ignore'):
            f_obs_sum = f_obs_float.sum(axis=axis)
            f_exp_sum = f_exp.sum(axis=axis)
            relative_diff = (np.abs(f_obs_sum - f_exp_sum) /
                             np.minimum(f_obs_sum, f_exp_sum))
            diff_gt_tol = (relative_diff > rtol).any()
        if diff_gt_tol:
            msg = (f"For each axis slice, the sum of the observed "
                   f"frequencies must agree with the sum of the "
                   f"expected frequencies to a relative tolerance "
                   f"of {rtol}, but the percent differences are:\n"
                   f"{relative_diff}")
            raise ValueError(msg)

    else:
        # Ignore 'invalid' errors so the edge case of a data set with length 0
        # is handled without spurious warnings.
        with np.errstate(invalid='ignore'):
            f_exp = f_obs.mean(axis=axis, keepdims=True)

    # `terms` is the array of terms that are summed along `axis` to create
    # the test statistic.  We use some specialized code for a few special
    # cases of lambda_.
    if lambda_ == 1:
        # Pearson's chi-squared statistic
        terms = (f_obs_float - f_exp)**2 / f_exp
    elif lambda_ == 0:
        # Log-likelihood ratio (i.e. G-test)
        terms = 2.0 * special.xlogy(f_obs, f_obs / f_exp)
    elif lambda_ == -1:
        # Modified log-likelihood ratio
        terms = 2.0 * special.xlogy(f_exp, f_exp / f_obs)
    else:
        # General Cressie-Read power divergence.
        terms = f_obs * ((f_obs / f_exp)**lambda_ - 1)
        terms /= 0.5 * lambda_ * (lambda_ + 1)

    stat = terms.sum(axis=axis)

    num_obs = _count(terms, axis=axis)
    ddof = asarray(ddof)
    p = chi2.sf(stat, num_obs - 1 - ddof)

    return Power_divergenceResult(stat, p)


def custom_chisquare(f_obs, f_exp=None, ddof=0, axis=0):
    """Calculate a one-way chi-square test.

    The chi-square test tests the null hypothesis that the categorical data
    has the given frequencies.

    Parameters
    ----------
    f_obs : array_like
        Observed frequencies in each category.
    f_exp : array_like, optional
        Expected frequencies in each category.  By default the categories are
        assumed to be equally likely.
    ddof : int, optional
        "Delta degrees of freedom": adjustment to the degrees of freedom
        for the p-value.  The p-value is computed using a chi-squared
        distribution with ``k - 1 - ddof`` degrees of freedom, where `k`
        is the number of observed frequencies.  The default value of `ddof`
        is 0.
    axis : int or None, optional
        The axis of the broadcast result of `f_obs` and `f_exp` along which to
        apply the test.  If axis is None, all values in `f_obs` are treated
        as a single data set.  Default is 0.

    Returns
    -------
    chisq : float or ndarray
        The chi-squared test statistic.  The value is a float if `axis` is
        None or `f_obs` and `f_exp` are 1-D.
    p : float or ndarray
        The p-value of the test.  The value is a float if `ddof` and the
        return value `chisq` are scalars.

    See Also
    --------
    scipy.stats.power_divergence
    scipy.stats.fisher_exact : Fisher exact test on a 2x2 contingency table.
    scipy.stats.barnard_exact : An unconditional exact test. An alternative
        to chi-squared test for small sample sizes.

    Notes
    -----
    This test is invalid when the observed or expected frequencies in each
    category are too small.  A typical rule is that all of the observed
    and expected frequencies should be at least 5. According to [3]_, the
    total number of samples is recommended to be greater than 13,
    otherwise exact tests (such as Barnard's Exact test) should be used
    because they do not overreject.

    Also, the sum of the observed and expected frequencies must be the same
    for the test to be valid; `chisquare` raises an error if the sums do not
    agree within a relative tolerance of ``1e-8``.

    The default degrees of freedom, k-1, are for the case when no parameters
    of the distribution are estimated. If p parameters are estimated by
    efficient maximum likelihood then the correct degrees of freedom are
    k-1-p. If the parameters are estimated in a different way, then the
    dof can be between k-1-p and k-1. However, it is also possible that
    the asymptotic distribution is not chi-square, in which case this test
    is not appropriate.

    References
    ----------
    .. [1] Lowry, Richard.  "Concepts and Applications of Inferential
           Statistics". Chapter 8.
           https://web.archive.org/web/20171022032306/http://vassarstats.net:80/textbook/ch8pt1.html
    .. [2] "Chi-squared test", https://en.wikipedia.org/wiki/Chi-squared_test
    .. [3] Pearson, Karl. "On the criterion that a given system of deviations from the probable
           in the case of a correlated system of variables is such that it can be reasonably
           supposed to have arisen from random sampling", Philosophical Magazine. Series 5. 50
           (1900), pp. 157-175.

    Examples
    --------
    When just `f_obs` is given, it is assumed that the expected frequencies
    are uniform and given by the mean of the observed frequencies.

    >>> from scipy.stats import chisquare
    >>> chisquare([16, 18, 16, 14, 12, 12])
    (2.0, 0.84914503608460956)

    With `f_exp` the expected frequencies can be given.

    >>> chisquare([16, 18, 16, 14, 12, 12], f_exp=[16, 16, 16, 16, 16, 8])
    (3.5, 0.62338762774958223)

    When `f_obs` is 2-D, by default the test is applied to each column.

    >>> obs = np.array([[16, 18, 16, 14, 12, 12], [32, 24, 16, 28, 20, 24]]).T
    >>> obs.shape
    (6, 2)
    >>> chisquare(obs)
    (array([ 2.        ,  6.66666667]), array([ 0.84914504,  0.24663415]))

    By setting ``axis=None``, the test is applied to all data in the array,
    which is equivalent to applying the test to the flattened array.

    >>> chisquare(obs, axis=None)
    (23.31034482758621, 0.015975692534127565)
    >>> chisquare(obs.ravel())
    (23.31034482758621, 0.015975692534127565)

    `ddof` is the change to make to the default degrees of freedom.

    >>> chisquare([16, 18, 16, 14, 12, 12], ddof=1)
    (2.0, 0.73575888234288467)

    The calculation of the p-values is done by broadcasting the
    chi-squared statistic with `ddof`.

    >>> chisquare([16, 18, 16, 14, 12, 12], ddof=[0,1,2])
    (2.0, array([ 0.84914504,  0.73575888,  0.5724067 ]))

    `f_obs` and `f_exp` are also broadcast.  In the following, `f_obs` has
    shape (6,) and `f_exp` has shape (2, 6), so the result of broadcasting
    `f_obs` and `f_exp` has shape (2, 6).  To compute the desired chi-squared
    statistics, we use ``axis=1``:

    >>> chisquare([16, 18, 16, 14, 12, 12],
    ...           f_exp=[[16, 16, 16, 16, 16, 8], [8, 20, 20, 16, 12, 12]],
    ...           axis=1)
    (array([ 3.5 ,  9.25]), array([ 0.62338763,  0.09949846]))

    """
    return power_divergence(f_obs, f_exp=f_exp, ddof=ddof, axis=axis,
                            lambda_="pearson")

###### DG ADDITION END ######



def wiener_like_contaminant(
    value,
    cont_x,
    v,
    sv,
    a,
    z,
    sz,
    t,
    st,
    t_min,
    t_max,
    err,
    n_st,
    n_sz,
    use_adaptive,
    simps_err,
):
    """Log-likelihood for the simple DDM including contaminants"""
    return hddm.wfpt.wiener_like_contaminant(
        value,
        cont_x.astype(np.int32),
        v,
        sv,
        a,
        z,
        sz,
        t,
        st,
        t_min,
        t_max,
        err,
        n_st,
        n_sz,
        use_adaptive,
        simps_err,
    )


WienerContaminant = stochastic_from_dist(
    name="Wiener Simple Diffusion Process", logp=wiener_like_contaminant
)


def general_WienerCont(err=1e-4, n_st=2, n_sz=2, use_adaptive=1, simps_err=1e-3):
    _like = lambda value, cont_x, v, sv, a, z, sz, t, st, t_min, t_max, err=err, n_st=n_st, n_sz=n_sz, use_adaptive=use_adaptive, simps_err=simps_err: wiener_like_contaminant(
        value,
        cont_x,
        v,
        sv,
        a,
        z,
        sz,
        t,
        st,
        t_min,
        t_max,
        err=err,
        n_st=n_st,
        n_sz=n_sz,
        use_adaptive=use_adaptive,
        simps_err=simps_err,
    )
    _like.__doc__ = wiener_like_contaminant.__doc__
    return stochastic_from_dist(name="Wiener Diffusion Contaminant Process", logp=_like)


def generate_wfpt_stochastic_class(
    wiener_params=None, sampling_method="cssm", cdf_range=(-5, 5), sampling_dt=1e-4
):
    """
    create a wfpt stochastic class by creating a pymc nodes and then adding quantile functions.

    :Arguments:
        wiener_params: dict <default=None>
            dictonary of wiener_params for wfpt likelihoods
        sampling_method: str <default='cssm'>
            an argument used by hddm.generate.gen_rts
        cdf_range: sequence <default=(-5,5)>
            an argument used by hddm.generate.gen_rts
        sampling_dt: float <default=1e-4>
            an argument used by hddm.generate.gen_rts

    :Output:
        wfpt: class
            the wfpt stochastic
    """

    # set wiener_params
    if wiener_params is None:
        wiener_params = {
            "err": 1e-4,
            "n_st": 2,
            "n_sz": 2,
            "use_adaptive": 1,
            "simps_err": 1e-3,
            "w_outlier": 0.1,
        }
    wp = wiener_params

    # create likelihood function
    def wfpt_like(x, v, sv, a, z, sz, t, st, p_outlier=0):
        if x["rt"].abs().max() < 998:
            return hddm.wfpt.wiener_like(
                x["rt"].values, v, sv, a, z, sz, t, st, p_outlier=p_outlier, **wp
            )
        else:  # for missing RTs. Currently undocumented.
            noresponse = x["rt"].abs() >= 999
            ## get sum of log p for trials with RTs as usual ##
            logp_resp = hddm.wfpt.wiener_like(
                x.loc[~noresponse, "rt"].values,
                v,
                sv,
                a,
                z,
                sz,
                t,
                st,
                p_outlier=p_outlier,
                **wp
            )

            # get number of no-response trials
            n_noresponse = sum(noresponse)
            k_upper = sum(x.loc[noresponse, "rt"] > 0)

            # percentage correct according to probability to get to upper boundary
            if v == 0:
                p_upper = z
            else:
                p_upper = (np.exp(-2 * a * z * v) - 1) / (np.exp(-2 * a * v) - 1)

            logp_noresp = stats.binom.logpmf(k_upper, n_noresponse, p_upper)
            return logp_resp + logp_noresp

    # create random function
    def random(
        self,
        keep_negative_responses=True,
        add_model=False,
        add_outliers=False,
        add_model_parameters=False,
        keep_subj_idx=False,
    ):
        # print(self.value)
        # print(type(self.value))
        assert sampling_method in [
            "cdf",
            "drift",
            "cssm",
        ], "Sampling method is invalid!"

        if sampling_method == "cdf" or sampling_method == "drift":
            return hddm.utils.flip_errors(
                hddm.generate.gen_rts(
                    method=sampling_method,
                    size=self.shape,
                    dt=sampling_dt,
                    range_=cdf_range,
                    structured=True,
                    **self.parents.value
                )
            )
        elif sampling_method == "cssm":
            keys_tmp = self.parents.value.keys()
            cnt = 0
            theta = np.zeros(len(list(keys_tmp)), dtype=np.float32)

            for param in model_config["full_ddm_vanilla"]["params"]:
                theta[cnt] = np.array(self.parents.value[param]).astype(np.float32)
                cnt += 1

            sim_out = simulator(
                theta=theta, model="full_ddm_vanilla", n_samples=self.shape[0], max_t=20
            )

            if add_outliers:
                if self.parents.value["p_outlier"] > 0.0:
                    sim_out = hddm_dataset_generators._add_outliers(
                        sim_out=sim_out,
                        p_outlier=self.parents.value["p_outlier"],
                        max_rt_outlier=1 / wiener_params["w_outlier"],
                    )

            sim_out_proc = hddm_preprocess(
                sim_out,
                keep_negative_responses=keep_negative_responses,
                keep_subj_idx=keep_subj_idx,
                add_model_parameters=add_model_parameters,
            )

            if add_model:
                if (
                    (self.parents.value["sz"] == 0)
                    and (self.parents.value["sv"] == 0)
                    and (self.parents.value["st"] == 0)
                ):
                    sim_out_proc["model"] = "ddm_vanilla"
                else:
                    sim_out_proc["model"] = "full_ddm_vanilla"

            sim_out_proc = hddm.utils.flip_errors(
                sim_out_proc
            )  # ['rt'] * sim_out_proc['response']

            return sim_out_proc

    # create pdf function
    def pdf(self, x):
        out = hddm.wfpt.pdf_array(x, **self.parents)
        return out

    # create cdf function
    def cdf(self, x):
        return hddm.cdfdif.dmat_cdf_array(x, w_outlier=wp["w_outlier"], **self.parents)

    # create wfpt class
    wfpt = stochastic_from_dist("wfpt", wfpt_like)

    # add pdf and cdf_vec to the class
    wfpt.pdf = pdf
    wfpt.cdf_vec = lambda self: hddm.wfpt.gen_cdf_using_pdf(
        time=cdf_range[1], **dict(list(self.parents.items()) + list(wp.items()))
    )
    wfpt.cdf = cdf
    wfpt.random = random

    # add quantiles functions
    add_quantiles_functions_to_pymc_class(wfpt)

    return wfpt


def add_quantiles_functions_to_pymc_class(pymc_class):
    """add quantiles methods to a pymc class.

    :Input:
        pymc_class: class
    """

    # turn pymc node into the final wfpt_node
    def compute_quantiles_stats(self, quantiles=(0.1, 0.3, 0.5, 0.7, 0.9)):
        """
        compute quantiles statistics
        Input:
            quantiles : sequence
                the sequence of quantiles,  e.g. (0.1, 0.3, 0.5, 0.7, 0.9)
        """
        try:
            if all(self._quantiles_edges == np.asarray(quantiles)):
                return
        except AttributeError:
            pass

        if hasattr(self, "_is_average_node"):
            raise AttributeError("cannot recompute stats of average model")

        self._quantiles_edges = np.asarray(quantiles)

        data = self.value

        if np.all(~np.isnan(data["rt"])):

            # get proportion of data fall between the quantiles
            quantiles = np.array(quantiles)
            pos_proportion = np.diff(
                np.concatenate((np.array([0.0]), quantiles, np.array([1.0])))
            )
            neg_proportion = pos_proportion[::-1]
            proportion = np.concatenate((neg_proportion[::-1], pos_proportion))
            self._n_samples = len(data)

            # extract empirical RT at the quantiles
            self._empirical_quantiles = hddm.utils.data_quantiles(data, quantiles)
            ub_emp_rt = self._empirical_quantiles[1]
            lb_emp_rt = -self._empirical_quantiles[0]
            self._emp_rt = np.concatenate((lb_emp_rt[::-1], np.array([0.0]), ub_emp_rt))

            # get frequency of observed values
            freq_obs = np.zeros(len(proportion))
            freq_obs[: len(quantiles) + 1] = sum(data.rt < 0) * neg_proportion
            freq_obs[len(quantiles) + 1 :] = sum(data.rt > 0) * pos_proportion
            self._freq_obs = freq_obs

        else:

            # get proportion of data fall between the quantiles
            quantiles = np.array(quantiles)
            pos_proportion = np.diff(
                np.concatenate((np.array([0.0]), quantiles, np.array([1.0])))
            )
            neg_proportion = np.array([1])
            proportion = np.concatenate((neg_proportion[::-1], pos_proportion))
            self._n_samples = len(data)

            # extract empirical RT at the quantiles
            self._empirical_quantiles = hddm.utils.data_quantiles(data, quantiles)
            ub_emp_rt = self._empirical_quantiles[1]
            lb_emp_rt = -self._empirical_quantiles[0]
            self._emp_rt = np.concatenate((np.array([0.0]), ub_emp_rt))

            # get frequency of observed values
            freq_obs = np.zeros(len(proportion))
            freq_obs[0] = sum(np.isnan(data.rt)) * neg_proportion
            freq_obs[1:] = sum(data.rt > 0) * pos_proportion
            self._freq_obs = freq_obs

    def set_quantiles_stats(self, quantiles, n_samples, emp_rt, freq_obs, p_upper):
        """
        set quantiles statistics (used when one do not to compute the statistics from the stochastic's value)
        """
        self._quantiles_edges = np.asarray(quantiles)
        self._n_samples = n_samples
        self._emp_rt = emp_rt
        self._freq_obs = freq_obs

        nq = len(quantiles)
        q_lower = -emp_rt[:nq][::-1]
        q_upper = emp_rt[nq + 1 :]
        self._empirical_quantiles = (q_lower, q_upper, p_upper)

    def get_quantiles_stats(self, quantiles=(0.1, 0.3, 0.5, 0.7, 0.9)):
        """
        get quantiles statistics (after they were computed using compute_quantiles_stats)
        """
        self.compute_quantiles_stats(quantiles)

        stats = {
            "n_samples": self._n_samples,
            "emp_rt": self._emp_rt,
            "freq_obs": self._freq_obs,
        }
        return stats

    def _get_theoretical_proportion(self):

        # get cdf
        cdf = self.cdf(self._emp_rt)

        # get probabilities associated with theoretical RT indices
        theo_cdf = np.concatenate((np.array([0.0]), cdf, np.array([1.0])))

        # theoretical porportion
        proportion = np.diff(theo_cdf)

        # make sure there is no zeros since it causes bugs later on
        epsi = 1e-6
        proportion[proportion <= epsi] = epsi
        return proportion

    def chisquare(self):
        """
        compute the chi-square statistic over the stocastic's value
        """
        try:
            theo_proportion = self._get_theoretical_proportion()
        except (ValueError, FloatingPointError):
            return np.inf
        freq_exp = theo_proportion * self._n_samples
        # score, _ = stats.chisquare(self._freq_obs, freq_exp)
        score, _ = custom_chisquare(self._freq_obs, freq_exp)

        return score

    def gsquare(self):
        """
        compute G^2 (likelihood chi-square) statistic over the stocastic's value
        Note:
         this does return the actual G^2, but G^2 up to a constant which depend on the data
        """
        try:
            theo_proportion = self._get_theoretical_proportion()
        except ValueError:
            return -np.inf
        return 2 * sum(self._freq_obs * np.log(theo_proportion))

    def empirical_quantiles(self, quantiles=(0.1, 0.3, 0.5, 0.7, 0.9)):
        """
        return the quantiles of the Stochastic's value
        Output:
            q_lower - lower boundary quantiles
            q_upper - upper_boundary_quantiles
            p_upper - probability of hitting the upper boundary
        """
        self.compute_quantiles_stats(quantiles)

        return self._empirical_quantiles

    def theoretical_quantiles(self, quantiles=(0.1, 0.3, 0.5, 0.7, 0.9)):
        """
        return the theoretical quantiles based on Stochastic's parents
        Output:
            q_lower - lower boundary quantiles
            q_upper - upper_boundary_quantiles
            p_upper - probability of hitting the upper boundary
        """

        quantiles = np.asarray(quantiles)
        # generate CDF
        x_lower, cdf_lower, x_upper, cdf_upper = hddm.wfpt.split_cdf(*self.cdf_vec())

        # extract theoretical RT indices
        lower_idx = np.searchsorted(cdf_lower, quantiles * cdf_lower[-1])
        upper_idx = np.searchsorted(cdf_upper, quantiles * cdf_upper[-1])

        q_lower = x_lower[lower_idx]
        q_upper = x_upper[upper_idx]
        p_upper = cdf_upper[-1]

        return (q_lower, q_upper, p_upper)

    pymc_class.compute_quantiles_stats = compute_quantiles_stats
    pymc_class.set_quantiles_stats = set_quantiles_stats
    pymc_class.get_quantiles_stats = get_quantiles_stats
    pymc_class.chisquare = chisquare
    pymc_class.gsquare = gsquare
    pymc_class._get_theoretical_proportion = _get_theoretical_proportion
    pymc_class.empirical_quantiles = empirical_quantiles
    pymc_class.theoretical_quantiles = theoretical_quantiles


# create default Wfpt class
Wfpt = generate_wfpt_stochastic_class()
