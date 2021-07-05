"""Least-square fit of experimental CARS spectra."""
from pathlib import Path
from functools import wraps, partial
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


@_ensureLmfit
def slit_fit(nu, spec, init_params=None, lineshape='sGaussian',
             power_factor=1., eval_only=False,
             save_fit=False, dir_save=None, file_name=None):
    r"""fitting the experimental slit function with a chosen lineshape

    .. attention::
        It is recommended to always look for initial parameters by using the
        `eval_only` option to roughly match the shape of the experimental slit
        function. The built-in initial parameters may not work for all the
        cases.

    Parameters
    ----------
    nu : 1d array of floats
        Spectral positions in [:math:`\mathrm{cm}^{-1}`].
    spec : 1d array of floats
        Experimentally obtained slit function, usually an isolated atomic line
        from a calibration lamp.
    init_params : dict, optional
        Initial fitting parameters for the lineshape. Refer to
        :mod:`carspy.convol_fcn.asym_Voigt` and
        :mod:`carspy.convol_fcn.asym_Gaussian` for the required arguments.
    lineshape : str, optional
        Type of the lineshape, by default 'sGaussian'. Choose between
        'sGaussian' and 'sVoigt'.
    eval_only : bool, optional
        If true, returns the evaluation with given initial parameters.
    save_fit : bool, optional
        If true, fit results will be saved in a pickle file.
    dir_save : str, optional
        If `save_fit` is true, a string to the desired directory for saving.
    file_name : str, optional
        If `save_fit` is true, specify the file name without file extension.

    Returns
    -------
    1d array
        If `eval_only` is true, the evaluation result is returned.

    """
    if lineshape == 'sGaussian':
        slit_func = partial(asym_Gaussian, power_factor=power_factor)
    elif lineshape == 'sVoigt':
        slit_func = partial(asym_Voigt, power_factor=power_factor)
    else:
        raise ValueError("Please choose between 'sGaussian' and 'sVoigt'")
    slit_func.__name__ = 'slit_func'

    fit_model = Model(slit_func, independent_vars='w')

    if init_params is None:
        init_params = (
            ('w0', 0, True),
            ('sigma', 2, True, 0.01, 10),
            ('k', 2, True, 0, 10),
            ('a_sigma', -0.8, True, -5, 5),
            ('a_k', 1, True, -5, 5),
            ('sigma_L_l', 0.1, True, 0.01, 5),
            ('sigma_L_h', 0.1, True, 0.01, 5),
            ('offset', 0, True, 0, 1)
        )

    params = fit_model.make_params()
    params.add_many(*init_params)
    if eval_only:
        return fit_model.eval(params, w=nu)
    else:
        fit_result = fit_model.fit((spec/spec.max())**power_factor, params,
                                   w=nu)
        report_fit(fit_result, show_correl=True, modelpars=params)
        fit_result.plot()

        if save_fit:
            if dir_save and file_name:
                path_write = Path(dir_save) / (file_name + '.pkl')
                pkl_dump(path_write, fit_result)


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
            slit : 'sVoigt'
                Choose between (asymmetric) 'sVoigt' and 'sGaussian' as the
                slit impulse response function, see the documentations
                for :mod:`carspy.convol_fcn.asym_Voigt` and
                :mod:`carspy.convol_fcn.asym_Gaussian`. 'sGaussian' will be
                deprecated in future updates.
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
            fit : 'custom'
                Type of build-in fitting setups: 'T_x' or 'custom'.

                - 'T_x': fitting variables related to the experimental setup
                  (e.g., spectrometer) are inheritted from an existing fit
                  and fixed. Only temperature and species concentrations are
                  by default allowed to vary.
                - 'custom': all fitting variables need to be provided before a
                  fit can process.
                See :mod:`carspy.cars_fit.CarsFit.ls_fit` for details.


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
                             'fit': 'custom'
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
                        param1, param2, param3, param4, param5, param6):
        r"""
        Synthesize a CARS spectrum based on the experimental spectral domain.

        Parameters
        ----------
        nu_expt : 1d array of floats
            The spectral axis determined in the experiment. This is used as
            the independent variable during the fit.
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
        param1, param2, param3, param4, param5, param6 : float
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
        elif self.fit_mode['slit'] == 'sVoigt':
            nu_slit = asym_Voigt(w=nu_f, w0=(nu_f[0]+nu_f[-1])/2,
                                 sigma=param1, k=param2,
                                 a_sigma=param3, a_k=param4,
                                 sigma_L_l=param5, sigma_L_h=param6,
                                 offset=0)
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
    def ls_fit(self, add_params=None, path_fit=None, show_fit=False,
               eval_only=False, **kwargs):
        """
        Fitting the experimental CARS spectrum.

        .. attention::
            The least-quare fit module ``lmfit`` is necessary for this method.
            Please be aware that certain Python versions may not be supported
            by ``lmfit``. For displaying the fit results ``matplotlib`` will be
            needed as well.

        Parameters
        ----------
        add_params : nested tuple, optional
            List of parameters controlling the fitting process. This option can
            be used to modify these initial parameters:

            .. code-block:: python

                (('temperature', 2000, True, 250, 3000),
                 ('del_Tv', 0, False),
                 ('x_mol', 0.6, x_mol_var, 0.2, 1.5),
                 ('nu_shift', fit_params['nu_shift'], False),
                 ('nu_stretch', fit_params['nu_stretch'], False),
                 ('pump_lw', fit_params['pump_lw'], False),
                 ('param1', fit_params['param1'], False),
                 ('param2', fit_params['param2'], False),
                 ('param3', fit_params['param3'], False),
                 ('param4', fit_params['param4'], False),
                 ('param5', fit_params['param5'], False),
                 ('param6', fit_params['param6'], False))
            Each element of the nested tuple has the following element in
            order:
                variable_name : str
                    All the arguments of
                    :mod:`carspy.cars_fit.CarsFit.cars_expt_synth`
                    are admissible variables except for the independent
                    variable `nu_expt`.
                initial_guess : float
                    Initial guess or fixed value set for this variable.
                variable : bool
                    Determine if the variable is fixed (False) or not (True)
                    during the fit.
                lower_bound : float
                    Lower boundary for the fitting variable. If not provided,
                    negative infinity will be assumed.
                upper_bound : float
                    Upper boundary for the fitting variable. If not provide,
                    positive infinity will be assumed.
            For more details refer to the documentation of ``lmfit.Model``.
        path_fit : str
            Path to the `.pkl` file of fitting result created by
            :mod:`carspy.cars_fit.CarsFit.save_fit`. This allows importing
            the fitting result of an existing spectrum, such that the inferred
            values of certain parameters (such as those related to the
            spectrometer) could be re-used in the next fit. A standard use case
            for this would be the fitting result of a room-temperature
            spectrum. This is needed if the `fit` in `fit_mode` of
            :mod:`carspy.cars_fit.CarsFit` is set to `T_x`.
        show_fit : bool, optional
            If True, the fitting results will be reported and plotted. This is
            done via built-in functions in ``lmfit``.
        """
        # general setup
        fit_model = Model(self.cars_expt_synth, independent_vars=['nu_expt'])
        initi_params = []
        # different fitting modes
        if self.fit_mode['fit'] == 'T_x':
            if path_fit is None:
                raise ValueError("Please provide path to a .pkl file "
                                 "containing the fitting result of a spectrum")
            if self.fit_mode['chem_eq']:
                x_mol_var = False
            else:
                x_mol_var = True
            fit_params = pkl_load(path_fit).params
            initi_params = (('temperature', 2000, True, 250, 3000),
                            ('del_Tv', 0, False),
                            ('x_mol', 0.6, x_mol_var, 0.2, 1.5),
                            ('nu_shift', fit_params['nu_shift'], False),
                            ('nu_stretch', fit_params['nu_stretch'], False),
                            ('pump_lw', fit_params['pump_lw'], False),
                            ('param1', fit_params['param1'], False),
                            ('param2', fit_params['param2'], False),
                            ('param3', fit_params['param3'], False),
                            ('param4', fit_params['param4'], False),
                            ('param5', fit_params['param5'], False),
                            ('param6', fit_params['param6'], False))

        if self.fit_mode['fit'] == 'custom':
            if add_params is None:
                raise ValueError("Please specify fitting parameters first "
                                 "using add_params")
        params = fit_model.make_params()
        params.add_many(*initi_params)
        if add_params is not None:
            params.add_many(*add_params)

        if eval_only:
            return fit_model.eval(params, nu_expt=self.nu)
        else:
            self.fit_result = fit_model.fit(np.nan_to_num(self.spec_cars),
                                            params,
                                            nu_expt=self.nu, **kwargs)

            if show_fit:
                report_fit(self.fit_result, show_correl=True, modelpars=params)
                self.fit_result.plot()

    def save_fit(self, dir_save, file_name):
        """
        Save the fitting results in a pickle file.

        Parameters
        ----------
        dir_save : path
            A valid local directory.
        file_name : str
            File name of the pickle file to be saved.
        """
        path_write = Path(dir_save) / (file_name + '.pkl')
        pkl_dump(path_write, self.fit_result)
