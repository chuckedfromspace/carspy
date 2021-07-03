"""Additional functions used within the CARSpy program."""
from functools import wraps
import pickle
import numpy as np


try:
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    _HAS_SCIPY_PYPLOT = True
except Exception:
    _HAS_SCIPY_PYPLOT = False


def _ensureModules(function):
    if _HAS_SCIPY_PYPLOT:
        @wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)
        wrapper.__doc__ = function.__doc__
        return wrapper

    def no_op(*args, **kwargs):
        _message = ("Scipy and Matplotlib modules are required to carry out "
                    "peak finding")
        raise Exception(_message)
    no_op.__doc__ = function.__doc__
    return no_op


try:
    import cantera as ct
    _HAS_CANTERA = True
except Exception:
    _HAS_CANTERA = False


def _ensureCantera(function):
    if _HAS_CANTERA:
        @wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)
        wrapper.__doc__ = function.__doc__
        return wrapper

    def no_op(*args, **kwargs):
        _message = ("Cantera module is required for calculating equilibrium "
                    "composition. Please install cantera first or specify "
                    "custom eq_func in signal_as() instead.")
        raise Exception(_message)
    no_op.__doc__ = function.__doc__
    return no_op


@_ensureCantera
def eq_comp(temperature, pressure, init_comp, valid_from=1200.):
    """
    Calculate equilibrium composition at given temperature and pressure.

    .. attention::
        This function is only intended as a "dummy" template for setting up
        custom equilibrium solvers with ``cantera``. Please be aware of the
        applicabilities and uncertainties of various kinetic mechanisms.

    Parameters
    ----------
    temperature : float
        Temperature in [K].
    pressure : 1, float, optional
        Pressure in bars.
    init_comp : dict
        Initial gas mole fractions in a dictionary.
    valid_from : float
        Temperature lower boundary for when the mechanism is valid, default is
        1200 [K]. This value is by no means valid for all cases.
    """
    products = init_comp.copy()
    if temperature > valid_from:
        gas = ct.Solution('gri30.xml')
        gas.TPX = temperature, pressure*1e5, init_comp
        gas.equilibrate('TP')
        products = gas.mole_fraction_dict()

    # remove small values
    products = {key: products[key] for key in products if products[key] > 1e-5}

    return products


def downsample(w, w_fine, spec_fine, mode='local_mean'):
    """Downsample a fine spectrum according to specified coarse spectral axis.

    Parameters
    ----------
    w : sorted 1-D array of floats
        Coarse spectral axis (must be sorted and evenly spaced).
    w_fine : sorted 1-D array of floats
        Fine spectral axis (must be sorted and evenly spaced).
    spec_fine : 1-D array of floats
        Spectrum with fine resolution, must be of the same size as w_fine.
    mode : str, optional
        Two modes to choose from: 'local-mean' or 'interp', by default
        'local_mean'.

    Returns
    -------
    1-D array of floats
        Downsampled spectrum of the same size as w.
    """
    downsampled = []
    if mode == 'interp':
        downsampled = np.interp(w, w_fine, spec_fine)
    elif mode == 'local_mean':
        # downsample scale
        hw = int((w[1] - w[0])/(w_fine[1] - w_fine[0])/2)
        # search for closest indices
        w_fine = np.array(w_fine)
        idx = np.searchsorted(w_fine, w)
        idx[w_fine[idx] - w > np.diff(w_fine).mean()*0.5] -= 1
        # take local average based on the downsample scale
        downsampled = np.mean(
            [spec_fine[idx-_step] for _step in range(-hw, hw+1)], axis=0)

    return downsampled


def comp_normalize(comp_dict, target=1.0):
    """Normalize gas composition saved in a dictionary.

    Parameters
    ----------
    comp_dict : dict
        Gas composition in the measurement volume stored in a dictionary.
    target : float, optional
        Normalization factor, by default 1.0.

    Returns
    -------
    dict
        Normalized gas composition stored in a dictionary.
    """
    raw = sum(comp_dict.values())
    factor = target/raw
    return {key: value*factor for key, value in comp_dict.items()}


@_ensureModules
def loc_lines(spec, height, w_peaks, peak_indices=None, inspect=True):
    """[summary]

    Parameters
    ----------
    spec : 1d array of floats
        Spectrum containing distinct peaks.
    height : float
        A threshold for the peaks. Chose a value between 0 and 1.
    peak_indices : list of integer
        List of indices of peaks to use for calculating conversion factor.
    w_peaks : 1d array of floats
        List of targeted peaks in Angstrom.
    inpsect : bool, optional
        If true, the peaks will be displayed on top of the original spectrum.
    """
    peaks, _ = find_peaks(spec/spec.max(), height=height)
    if inspect:  # Inspect the peak finding results
        plt.plot(spec, 'k')
        plt.plot(peaks, spec[peaks], 'ro')
        plt.title('Close this window to continue the program')
        plt.show()

    if len(peaks) != len(w_peaks):
        raise ValueError('Adjust height to find the correct number \
                            of peaks (%d)' % (len(w_peaks),))

    if peak_indices is None:
        peak_indices = range(len(w_peaks))
    elif len(peaks) < len(peak_indices):
        raise ValueError('Number of indices exceed the peaks found.')

    conv_factor = (np.diff(w_peaks[peak_indices])
                   / np.diff(peaks[peak_indices]))
    offset = (w_peaks - peaks*conv_factor.mean())[peak_indices]
    # calculate spectral domain in nm
    w = (np.arange(len(spec))*conv_factor.mean() + offset.mean())/10

    return w, conv_factor, offset, peaks


def pkl_dump(path_write, data):
    """Dump data into a pickle file.

    Parameters
    ----------
    path_write : path
        Absolute path to the pickle file to be created.
    data: python object
    """
    with open(path_write, 'wb') as pf:
        pickle.dump(data, pf)


def pkl_load(path_load):
    """Load a data into from pickle file.

    Parameters
    ----------
    dir_save : path
        Absolute path to the pickle file to be loaded.
    """
    with open(path_load, 'rb') as pf:
        data = pickle.load(pf)

    return data
