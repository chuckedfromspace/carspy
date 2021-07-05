"""Functions used in the convolution of CARS spectrum.

- Laser lineshape
- Impulse spectral response function (ISRF) for the spectrometer slit
"""
import numpy as np


def gaussian_line(w, w0, sigma):
    """Generate a normalized Gaussian lineshape (integral equals to 1).

    Parameters
    ----------
    w : sorted 1d array of floats
        Spectral positions in wavenumber cm^(-1).
    w0 : float
        Center of the Gaussian lineshape in wavenumber cm^(-1).
    sigma : float
        FWHM of the Gaussian lineshape wavenumber cm^(-1).

    Returns
    -------
    1d array of floats
        Intensities of the normalized Gaussian lineshape over w.
    """
    _lineshape = 2/sigma*(np.log(2)/np.pi)**0.5*np.exp(
        -4*np.log(2)*((w-w0)/sigma)**2)

    return _lineshape


def lorentz_line(w, w0, sigma):
    """Generate a normalized Lorentzian lineshape (integral equals to 1).

    Parameters
    ----------
    w : sorted 1d array of floats
        Spectral positions in wavenumber cm^(-1).
    w0 : float
        Center of the Lorentzian lineshape in wavenumber cm^(-1).
    sigma : float
        FWHM of the Lorentzian lineshape wavenumber cm^(-1).

    Returns
    -------
    1d array of floats
        Intensities of the normalized Lorentzian lineshape over w.
    """
    _lineshape = 1/np.pi*(sigma/2)/((w-w0)**2+sigma**2/4)

    return _lineshape


def voigt_line(w, w0, sigma_V, sigma_L):
    """Generate an approximated Voigt lineshape following :cite:`Whiting:68`.

    Parameters
    ----------
    w : 1d array of floats
        Spectral positions in wavenumber cm^(-1).
    w0 : float
        Center of the Lorentzian lineshape in wavenumber cm^(-1).
    sigma_V : float
        FWHM of the Voigt lineshape wavenumber cm^(-1).
    sigma_L : float
        FWHM of the Lorentzian lineshape wavenumber cm^(-1).

    Returns
    -------
    1d array
        Intensities of the Voigt lineshape over w.
    """
    # Preparations
    _ratio = sigma_L/sigma_V
    I_g = 1/(sigma_V*(1.065 + 0.447*_ratio + 0.058*_ratio**2))
    _w = abs(w-w0)/sigma_V
    # Building up the function
    _term_1 = I_g*(1-_ratio)*np.exp(-2.772*_w**2) + _ratio/(1 + 4*_w**2)
    _term_2 = 0.016*(1-_ratio)*_ratio*(np.exp(-0.4*_w**2.25)
                                       - 10/(10 + _w**2.25))

    return _term_1 + _term_2


def asym_Gaussian(w, w0, sigma, k, a_sigma, a_k, offset, power_factor=1.):
    """Asymmetric super-Gaussian following :cite:`Beirle:17`.

    Parameters
    ----------
    w : sorted 1d array of floats
        Spectral positions in wavenumber cm^(-1).
    w0 : float
        Center of the asymmetric Gaussian function in wavenumber cm^(-1).
    sigma : float
        FWHM of the Gaussian function in wavenumber cm^(-1).
    k : float
        Controls the skewing of the asymmetry.
    a_sigma, a_k : float
        Tuning factors for sigma and k.
    offset : float
        Background offset (from experimental spectrum).
    power_factor : float
        Power factor on the output (e.g. can be used during slit fitting).

    Returns
    -------
    1d array of floats
        Intensities of the peak-normalized asymmetric super-Gaussian over w.
    """
    response_low = np.exp(-abs((w[w <= w0]-w0)/(sigma-a_sigma))**(k-a_k))
    response_high = np.exp(-abs((w[w > w0]-w0)/(sigma+a_sigma))**(k+a_k))
    response = (np.append(response_low, response_high) + offset)**power_factor

    return np.nan_to_num(response/response.max())


def asym_Voigt(w, w0, sigma, k, a_sigma, a_k, sigma_L_l, sigma_L_h, offset,
               power_factor=1.):
    """Asymmetric super-Voigt.

    .. note::

        This is based on the super-Gaussian from :cite:`Beirle:17`, with
        additional convolution with two Lorentzian profiles to better capture
        slow-decaying wings in some experimental slit function

    Parameters
    ----------
    w : sorted 1d array of floats
        Spectral positions in wavenumber cm^(-1).
    w0 : float
        Center of the asymmetric Gaussian function in wavenumber cm^(-1).
    sigma : float
        FWHM of the Gaussian function in wavenumber cm^(-1).
    k : float
        Controls the skewing of the asymmetry.
    a_sigma, a_k : float
        Tuning factors for sigma and k.
    sigma_L_l : float
        FWHM of the Lorentzian function in wavenumber cm^(-1) for the
        lower half.
    sigma_L_h : float
        FWHM of the Lorentzian function in wavenumber cm^(-1) for the
        higher half.
    offset : float
        Background offset.
    power_factor : float
        Power factor on the output (e.g. can be used during slit fitting).

    Returns
    -------
    1d array of floats
        Intensities of the peak-normalized asymmetric super-Gaussian over w.
    """
    response_low = np.exp(-abs((w-w0)/(sigma-a_sigma))**(k-a_k))
    response_high = np.exp(-abs((w-w0)/(sigma+a_sigma))**(k+a_k))
    response_low = np.convolve(response_low,
                               lorentz_line(w, (w[0]+w[-1])/2, sigma_L_l),
                               'same')
    response_high = np.convolve(response_high,
                                lorentz_line(w, (w[0]+w[-1])/2, sigma_L_h),
                                'same')
    response_low = response_low/response_low.max()
    response_high = response_high/response_high.max()
    response = (np.append(response_low[np.where(w <= w0)],
                response_high[np.where(w > w0)]) + offset)**power_factor
    return np.nan_to_num(response/response.max())


def asym_Voigt_deprecated(w, w0, sigma_V_l, sigma_V_h, sigma_L_l, sigma_L_h,
                          offset):
    """Asymmetric Voigt profile following NRC.

    .. admonition:: Deprecated
       :class: attention

       This profile cannot capture certain slit functions with broadened
       Gaussian profile.

    Parameters
    ----------
    w : sorted 1d array of floats
        Spectral positions in wavenumber cm^(-1).
    w0 : float
        Center of the asymmetric Gaussian function in wavenumber cm^(-1).
    sigma_V_l : float
        FWHM of the Voigt function in wavenumber cm^(-1) for the lower half.
    sigma_V_h : float
        FWHM of the Voigt function in wavenumber cm^(-1) for the higher half.
    sigma_L_l : float
        FWHM of the Lorentzian function in wavenumber cm^(-1) for the
        lower half.
    sigma_L_h : float
        FWHM of the Lorentzian function in wavenumber cm^(-1) for the
        higher half.
    offset : float
        Background offset.

    Returns
    -------
    1d array of floats
        Intensities of the peak-normalized asymmetric super-Gaussian over w.
    """
    response_low = voigt_line(w[w <= w0], w0, sigma_V_l, sigma_L_l)
    response_high = voigt_line(w[w > w0], w0, sigma_V_h, sigma_L_h)
    response = (np.append(response_low/response_low.max(),
                          response_high/response_high.max()) +
                offset)

    return response/response.max()


def slit_ISRF(w, w0, param_1, param_2, param_3, param_4, param_5, param_6,
              offset, mode='sGaussian'):
    """Impulse spectral response function (ISRF) as the slit function.

    Parameters
    ----------
    w : sorted 1d array of floats
        Spectral positions in wavenumber cm^(-1).
    w0 : float
        Center of the asymmetric Gaussian function in wavenumber cm^(-1).
    param_1, param_2, param_3, param_4 : float
        Parameters needed for the asymmetric ISRF depending on the mode.

        - 'sGaussian':
            sigma : float
                FWHM of the Gaussian function in wavenumber cm^(-1).
            k : float
                Controls the skewing of the asymmetry.
            a_sigma, a_k : float
                Tuning factors for sigma and k.
        - 'Voigt':
            sigma_V_l : float
                FWHM of the Voigt function in wavenumber cm^(-1) for
                the lower half.
            sigma_L_l : float
                FWHM of the Lorentzian function in wavenumber cm^(-1) for
                the lower half.
            sigma_V_h : float
                FWHM of the Voigt function in wavenumber cm^(-1) for
                the higher half.
            sigma_L_h : float
                FWHM of the Lorentzian function in wavenumber cm^(-1) for
                the higher half.
    offset : float
        Background offset.
    mode : 'sGaussian', str, optional
        Two options for the ISRF:

        - Asymmetric super Gaussian: 'sGaussian'.
        - Asymmetric Voigt: 'Voigt'.

    Returns
    -------
    1d array of floats
        Intensities of the peak-normalized asymmetric ISRF.
    """
    slit_fc = []
    if mode == 'sGaussian':
        slit_fc = asym_Gaussian(w, w0, param_1, param_2, param_3,
                                param_4, offset)
    elif mode == 'Voigt':
        slit_fc = asym_Voigt(w, w0, param_1, param_2, param_3, param_4,
                             param_5, param_6, offset)

    return slit_fc
