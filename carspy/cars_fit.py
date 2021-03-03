"""Least-square fit of experimental CARS spectra."""
from pathlib import Path
from functools import wraps
import numpy as np
from .cars_synth import CarsSpectrum
from .convol_fcn import asym_Gaussian, asym_Voigt
from .utils import pkl_dump, pkl_load, downsample

try:
    from lmfit import Model
    from lmfit.printfuncs import report_fit
    _HAS_LMFIT = True
except Exception:
    _HAS_LMFIT = False


def _ensureLmfit(function):
    if _HAS_LMFIT:
        @wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)
        wrapper.__doc__ = function.__doc__
        return wrapper

    def no_op(*args, **kwargs):
        _message = ("lmfit module is required for carrying out the built-in "
                    "least-square fit. Please install lmfit first or build "
                    "custom fit routines based on cars_expt_sync().")
        raise Exception(_message)
    no_op.__doc__ = function.__doc__
    return no_op


def bg_removal(spec, bg=None):
    """Subtract background from spectrum and normalize it by its peak.

    Parameters
    ----------
    spec : 1d array
        Spectrum.
    bg : None, 1d array, optional
        Background noise.

    Return
    ------
    id array
        Background-subtracted, peak-normalized spectrum.
    """
    if bg is not None:
        _spec = (np.array(spec) - np.array(bg)).flatten()
    else:
        _spec = np.array(spec).flatten()

    return _spec/_spec.max()


class CarsFit():
    """Fitting experimental CARS spectrum.

    .. note::
        It can also be used to fit the laser linewidth or slit function.

    """

    def __init__(self, spec_cars, nu_spec, ref_fac=100., bg_cars=None,
                 spec_stokes=None, bg_stokes=None, fit_mode=None, **kwargs):
        r"""Input measured CARS spectrum and its background.

        Parameters
        ----------
        spec_cars : 1d array of floats
            Measured CARS spectrum.
        nu_spec : 1d array of floats
            Spectral range (axis) of the supplied CARS spectrum in
            [:math:`\mathrm{cm}^{-1}`]. This can either be absolute values or
            Raman shifted ones.
        ref_fac : float
            Refining factor based on the supplied spectral domain, by default
            100. This factor is used in the fitting procedure to synthesize
            spectra with much higher spectral resolution to be compared (after
            downsampling) with the supplied (expt.) CARS spectrum. Higher value
            will result in more accurate spectra at the cost of CPU time.
        bg_cars : 1d array of floats
            Background noise for the measured CARS spectrum.
        spec_stokes : 1d array of floats
            Measured broadband Stokes profile (usually with Ar), by default
            None. If not supplied, it is assumed that `spec_cars` is already
            corrected by the Stokes profile.
        bg_stokes : 1d array of floats
            Background noise for the measured Stokes profile.
        fit_mode : dict, optional
            A dictionary containing the control parameters used to perform the
            CARS fit. By default:

            power_factor : 0
                0 (fit I) or 1 (fit sqrt(I)).
            downsample : 'local_mean'
                Choose between 'local_mean' (highly efficient custom algorithm)
                and 'interp' (interpolation with :mod:`numpy.interp`).
            slit : 'Voigt'
                Choose between (asymmetric) 'Voigt' and 'sGaussian' as the slit
                impulse response function, see the documentations
                for :mod:`carspy.convol_fcn.asym_Voigt` and
                :mod:`carspy.convol_fcn.asym_Gaussian`.
            pump_ls : 'Gaussian'
                'Gaussian' or 'Lorentzian' laser lineshape.
            chi_rs : 'G-matrix'
                Choose between 'isolated' and 'G-matrix' for the consideration
                of pressure effects.
            convol : 'Kataoka'
                'Kataoka'/'K' (double convolution) or 'Yuratich'/'Y' (single
                convolution).
            doppler_effect : True
                Whether or not to consider Doppler effect on the Raman
                lineshape.
            chem_eq : False
                Whether or not to assume chemical equilibrium during the fit.
            fit : 'T_fit'
                Type of build-in fitting setups: 'room_fit' or 'T_fit' for room
                spectrum (for fitting slit function or laser linewidth) and
                normal spectrum (with fixed slit/laser lineshape).


        Other Parameters
        ----------------
        **kwargs:
            This method also allows the keyword arguments found for
            initializing :mod:`carspy.cars_synth.CarsSpectrum`.
        """
        # settings for the fit
        if fit_mode is None:
            self.fit_mode = {'power_factor': 0,
                             'downsample': 'local_mean',
                             'slit': 'Voigt',
                             'pump_ls': 'Gaussian',
                             'chi_rs': 'G-matrix',
                             'convol': 'Kataoka',
                             'doppler_effect': True,
                             'chem_eq': False,
                             'fit': 'room_fit'
                             }
        else:
            self.fit_mode = fit_mode

        # create subset of synth mode for CarsSpectrum
        self.synth_mode = {'pump_ls': self.fit_mode['pump_ls'],
                           'chi_rs': self.fit_mode['chi_rs'],
                           'convol': self.fit_mode['convol'],
                           'doppler_effect': self.fit_mode['doppler_effect'],
                           'chem_eq': self.fit_mode['chem_eq'],
                           }

        # subtract background (optional) and normalize by max
        self.spec_cars = bg_removal(spec_cars, bg_cars)
        if spec_stokes is not None:
            self.spec_stokes = bg_removal(spec_stokes, bg_stokes)
            if len(self.spec_stokes) != len(self.spec_cars):
                raise ValueError("The length of spec_cars needs to be "
                                 "identical to that of spec_stok")
        else:
            self.spec_stokes = spec_stokes

        # fixed properties
        self.nu = nu_spec
        self.ref_fac = ref_fac

        if len(self.nu) != len(self.spec_cars):
            raise ValueError("The length of spec_cars needs to be identical "
                             "to that of nu_spec")

        # setup CarsSpectrum
        self.spec_synth = CarsSpectrum(**kwargs)

        # define fit result
        self.fit_result = []

    def preprocess(self, w_Stokes=0, nu_Stokes=0, crop=None,
                   bg_subtract=False, bg_offset=0, bg_loc=None):
        r"""Prepare the raw data for the fitting.

        Parameters
        ----------
        w_Stokes : float, optional
            Center wavelength of the Stokes (e.g., dye laser) beam in [nm], by
            default 0.
        nu_shift : float, optional
            Center wavenumber of the Stokes beam in [:math:`\mathrm{cm}^{-1}`],
            by default 0.
            In essence, `w_Stokes` and `nu_Stokes` are equivalent.
        crop : list, optional
            Two indices to crop the spectrum with, by default None. Needs to be
            adjusted based on the experimental setup.
        bg_subtract : bool, optional
            If True, an extra offset specified by `bg_offset` or determined
            within `bg_loc` is subtracted from the spectrum.
            This is not recommended as there shouldn't be any physical
            background left if backgrounds are subtracted properly from the
            experimental spectrum beforehand. This might help if S/N is bad.
            By default it is set to False.
        bg_offset : float, optional
            Value used as background to subtract, be default 0. This is ignored
            if `bg_log` is provided.
        bg_loc : list, optional
            Two indices to select the part of spectrum as background, only used
            if `bg_subtract` is True.
        """
        # remove very small values in the argon spectrum
        if self.spec_stokes is not None:
            self.spec_stokes[self.spec_stokes <= 1e-3] = 1e-3
            self.spec_cars = self.spec_cars / self.spec_stokes

        # extra background removal (USE WITH CAUTION)
        if bg_subtract:
            if bg_loc is not None:
                _bg = self.spec_cars[bg_loc[0]:bg_loc[1]].mean()
            else:
                _bg = bg_offset
            self.spec_cars = self.spec_cars - _bg

            self.spec_cars[self.spec_cars < 0] = 0

        # crop signal and spectral axis and re-normalize
        if crop is not None:
            self.spec_cars = self.spec_cars[crop[0]:crop[1]]
            self.nu = self.nu[crop[0]:crop[1]]
        # take the square root if 'power_factor' is 1
        self.spec_cars = self.spec_cars**(0.5**self.fit_mode['power_factor'])
        self.spec_cars = self.spec_cars/self.spec_cars.max()

        # convert the spectral axis to relative wavenumber to match
        # the CARS program
        if w_Stokes != 0:
            self.nu = self.nu - 1e7/w_Stokes
        else:
            self.nu = self.nu - nu_Stokes

    def cars_expt_synth(self, nu_expt, x_mol, temperature, del_Tv, nu_shift,
                        nu_stretch, pump_lw,
                        param1, param2, param3, param4):
        r"""
        Synthesize a CARS spectrum based on the experimental spectral domain.

        Parameters
        ----------
        x_mol : float
            Mole fraction of probed molecule.
        temperature : float
            Temperature in the probe volume in [K].
        nu_shift : float
            Shift applied to correctly center the spectrum in
            [:math:`\mathrm{cm}^{-1}`].
        nu_strech : float
            Strech applied to nu to compensate for incorrect dispersion
            calibration.
        pump_lw : float
            Pump laser linewdith in [:math:`\mathrm{cm}^{-1}`].
        del_Tv : float
            The amount vibrational temperature exceeds the rotational
            temperature.
        param1, param2, param3, param4 : float
            Fitting parameters for the slit function

        Return
        ------
        1d array
            Syntheiszed and downsampled CARS spectrum to match the length and
            spectral resolution of the measured spectrum. The spectrum is
            normalized by its peak value.
        """
        # shift the experimental spectral axis and refine it
        nu_expt = nu_expt*nu_stretch + nu_shift
        # using magnification to refine the grid
        _fine_factor = self.ref_fac
        _del_nu = nu_expt[1] - nu_expt[0]
        _nu_expt_pad = np.pad(nu_expt, (5, 5), 'reflect', reflect_type='odd')
        nu_f = np.arange(start=_nu_expt_pad[0], stop=_nu_expt_pad[-1],
                         step=_del_nu/_fine_factor)

        # calculate the slit function
        nu_slit = []
        if self.fit_mode['slit'] == 'sGaussian':
            nu_slit = asym_Gaussian(w=nu_f, w0=(nu_f[0]+nu_f[-1])/2,
                                    sigma=param1, k=param2,
                                    a_sigma=param3, a_k=param4, offset=0)
        elif self.fit_mode['slit'] == 'Voigt':
            nu_slit = asym_Voigt(w=nu_f, w0=(nu_f[0]+nu_f[-1])/2,
                                 sigma_V_l=param1,
                                 sigma_V_h=param2, sigma_L_l=param3,
                                 sigma_L_h=param4, offset=0)
        # calculate the CARS spectrum
        _, I_as = self.spec_synth.signal_as(x_mol=x_mol,
                                            temperature=temperature, nu_s=nu_f,
                                            pump_lw=pump_lw, del_Tv=del_Tv,
                                            synth_mode=self.synth_mode)
        # convolute the slit function with the CARS spectrum
        I_as = np.convolve(I_as, nu_slit, 'same')

        # downsampling to the experimental spectral grid
        I_as_down = downsample(nu_expt, nu_f, I_as,
                               mode=self.fit_mode['downsample'])**(
                                   0.5**self.fit_mode['power_factor'])
        return np.nan_to_num(I_as_down/I_as_down.max())

    @_ensureLmfit
    def ls_fit(self, T_0=None, pump_lw=None, path_room_fit=None,
               show_fit=False, add_params=None):
        """
        Fitting the experimental CARS spectrum.

        Parameters
        ----------
        show_fit : False, bool, optional
            If True, the fitting results will be reported and plotted
            as per lmfit.
        """
        # general setup
        fit_model = Model(self.cars_expt_synth, independent_vars=['nu_expt'])
        initi_params = []

        # different fitting modes
        if self.fit_mode['fit'] == 'room_fit':
            # fit slit and pump together at room T
            if None in (T_0, pump_lw):
                raise ValueError(('Please provide measured room temperature'
                                  'and pump linewidth in ls_fit()'))
            initi_params = (('temperature', T_0, False),
                            ('del_Tv', 0, False, 0, 300),
                            ('x_mol', 0.79, False),
                            ('nu_shift', 0, True, -3, 3),
                            ('nu_stretch', 1, False, 0.5, 1.5),
                            ('pump_lw', pump_lw, False, 0.1, 1),
                            ('param1', 8, True, 0.01, 20),
                            ('param2', 8, True, 0, 20),
                            ('param3', 1, True, 0, 20),
                            ('param4', 1, True, 0, 20))

        if self.fit_mode['fit'] == 'T_fit':
            if path_room_fit is None:
                raise ValueError('Please provide path to (room_fit).pkl')
            if self.fit_mode['chem_eq']:
                x_mol_var = False
            else:
                x_mol_var = True
            fit_params = pkl_load(path_room_fit).params
            initi_params = (('temperature', 2000, True, 250, 3000),
                            ('del_Tv', 0, False, 0, 500),
                            ('x_mol', 0.6, x_mol_var, 0.2, 1.5),
                            ('nu_shift', fit_params['nu_shift'], False, -3, 3),
                            ('nu_stretch', 1, False, 0.5, 1.5),
                            ('pump_lw', fit_params['pump_lw'], False, 0.1, 10),
                            ('param1', fit_params['param1'], False),
                            ('param2', fit_params['param2'], False),
                            ('param3', fit_params['param3'], False),
                            ('param4', fit_params['param4'], False))

        if self.fit_mode['fit'] == 'custom':
            initi_params = (('temperature', 2000, True, 250, 3000), )

        params = fit_model.make_params()
        params.add_many(*initi_params)
        if add_params is not None:
            params.add_many(*add_params)
        self.fit_result = fit_model.fit(np.nan_to_num(self.spec_cars), params,
                                        nu_expt=self.nu)

        if show_fit:
            report_fit(self.fit_result, show_correl=True, modelpars=params)
            self.fit_result.plot()

    def save_fit(self, dir_save, file_name='room_fit'):
        """
        Save the fitting results in a pickle file.

        Parameters
        ----------
        dir_save : str
            Path for saving the data.
        file_name : 'room_fit', str, optional
            Name of the file to be saved.
        """
        path_write = Path(dir_save) / (file_name + '.pkl')
        pkl_dump(path_write, self.fit_result)
