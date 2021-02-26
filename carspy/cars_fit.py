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
        return wrapper

    def no_op(*args, **kwargs):
        _message = ("lmfit module is required for carrying out the built-in "
                    "least-square fit. Please install lmfit first or build "
                    "custom fit routines based on cars_expt_sync().")
        raise Exception(_message)
    return no_op


def bg_removal(spec, bg=None, shots=1):
    """Subtract background from spectrum and normalize it by its peak.

    Parameters
    ----------
    spec : 1d array
        Spectrum.
    bg : None, 1d array, optional
        Background noise.
    shots : 1, int, optional
        Number of on CCD accumulated shots in the background. This is used for
        removing background in single-shot measurements.

    Return
    ------
    id array
        Background-subtracted, peak-normalized spectrum.
    """
    if bg is not None:
        _spec = (spec - bg/shots).flatten()
    else:
        _spec = spec.flatten()

    return _spec/_spec.max()


class CarsFit():
    """Fitting experimental CARS spectrum.

    It can also be used to fit the laser linewidth or slit function.
    """
    def __init__(self, spec_cars, spec_argon, nu_cal,
                 bg_cars=None, bg_argon=None, modes=None, **kwargs):
        """
        Input measured CARS spectrum and its background.

        Parameters
        ----------
        spec_cars : 1d array
            Measured CARS spectrum.
        bg_cars : 1d array
            Background noise taken when the lasers are blocked, with the same
            optical alignment and the same camera setting (gain, gate).
        spec_argon : 1d array
            Measured Argon nonresonant spectrum for the correction of
            Stokes profile.
        bg_argon : 1d array
            Background noise taken when the lasers are blocked, with the same
            optical alignment and the same camera setting (gain, gate).
        modes : None, dict, optional
            A dictionary containing the modes used to perform the CARS fit.

        Other Parameters
        ----------------
        **kwargs:
            This method also allows the keyword arguments found for
            initializing ``CarsSpectrum``.
        """
        # settings for the fit
        if modes is None:
            self.modes = {'power_factor': 0,       # 1: fit sqrt(I)
                          'downsample': 'local_mean',  # 'interp'
                          'slit': 'Voigt',         # 'sGaussian'
                          'pump_ls': 'Gaussian',   # 'Lorentzian'
                          'chi_rs': 'G-matrix',    # 'isolated'
                          'convol': 'Kataoka',     # 'Yuratich'
                          'Doppler': True,         # Doppler broadening
                          'chem_eq': False,
                          'fit': 'room_fit',       # 'room_pump_fit', 'T_fit'
                          'shots': 1               # on-CCD accumulation
                          }
        else:
            self.modes = modes

        # preprocessing the spectra
        self.spec_cars = bg_removal(spec_cars, bg_cars,
                                    shots=self.modes['shots'])
        self.spec_argon = bg_removal(spec_argon, bg_argon)

        # load the calibration results and extract the slit function
        self.nu = nu_cal

        if len(self.nu) != len(self.spec_cars):
            raise ValueError('The length of spec_cars needs to be identical'
                             'to that of the calibration spectrum')

        # setup CarsSpectrum
        self.spec_synth = CarsSpectrum(**kwargs)

        # define fit result
        self.fit_result = []

    def preprocess(self, w_Stokes=592.1, nu_shift=0, crop=(140, 210),
                   bg_subtract=False, bg_loc=(215, 225)):
        """
        Prepare the raw data for the fitting.

        Parameters
        ----------
        w_Stokes : 591, float, optional
            Center wavelength of the Stokes (e.g., dye laser) beam in nm.
        nu_shift : 35.4, float, optional
            A compensation to center the measured CARS spectrum correctly on a
            relative spectral axis. This is determined from fitting room
            temperature spectrum. [cm^-1]
        crop : (145, 210), list, optional
            Indices to crop the spectrum with. Needs to be adjusted based on
            the experimental setup.
        bg_subtract : False, bool, optional
            If True, an extra offset determined within bg_loc is subtracted
            from the spectrum.
        bg_loc : (215, 225), list, optional
            Indices to select the part of spectrum as background, only used if
            bg_subtract is true. This is however not recommended as there
            shouldn't be any physical background left if backgrounds are
            subtracted already.
        """
        # remove very small values in the argon spectrum
        self.spec_argon[self.spec_argon <= 1e-3] = 1e-3
        self.spec_cars = self.spec_cars / self.spec_argon

        # extra background removal; The factor of 0.9 is purely empirical
        _bg = self.spec_cars[bg_loc[0]:bg_loc[1]].mean()
        if _bg > 0.008 or bg_subtract:
            self.spec_cars = self.spec_cars - _bg*0.6

        self.spec_cars[self.spec_cars < 0] = 0
        # crop signal and spectral axis and re-normalize
        self.spec_cars = self.spec_cars[crop[0]:crop[1]]**(
            0.5**self.modes['power_factor'])
        self.nu = self.nu[crop[0]:crop[1]]
        self.spec_cars = self.spec_cars/self.spec_cars.max()

        # convert the spectral axis to relative wavenumber to match
        # the CARS program
        self.nu = self.nu - 1e7/w_Stokes + nu_shift

    def cars_expt_synth(self, nu_expt, x_mol, temperature, del_Tv, nu_shift,
                        nu_stretch, pump_lw,
                        param1, param2, param3, param4):
        """
        Synthesize a CARS spectrum based on the experimental spectral domain.

        Parameters
        ----------
        x_mol : float
            Mole fraction of probed molecule within [0, 1].
        temperature : float
            Temperature in the probe volume.
        nu_expt : 1d array
            Spectral axis of the experimental data.
        nu_shift : float
            Shift applied to nu to correctly center the spectrum.
        nu_strech : float
            Strech applied to nu to compensate for incorrect dispersion
            calibration.
        pump_lw : float
            Pump laser linewdith in cm^-1.
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
        _fine_factor = 140
        _del_nu = nu_expt[1] - nu_expt[0]
        _nu_expt_pad = np.pad(nu_expt, (5, 5), 'reflect', reflect_type='odd')
        nu_f = np.arange(start=_nu_expt_pad[0], stop=_nu_expt_pad[-1],
                         step=_del_nu/_fine_factor)

        # calculate the slit function
        nu_slit = []
        if self.modes['slit'] == 'sGaussian':
            nu_slit = asym_Gaussian(w=nu_f, w0=(nu_f[0]+nu_f[-1])/2,
                                    sigma=param1, k=param2,
                                    a_sigma=param3, a_k=param4, offset=0)
        elif self.modes['slit'] == 'Voigt':
            nu_slit = asym_Voigt(w=nu_f, w0=(nu_f[0]+nu_f[-1])/2,
                                 sigma_V_l=param1,
                                 sigma_V_h=param2, sigma_L_l=param3,
                                 sigma_L_h=param4, offset=0)
        # calculate the CARS spectrum
        _, I_as = self.spec_synth.signal_as(x_mol=x_mol,
                                            temperature=temperature, nu_s=nu_f,
                                            pump_lw=pump_lw, del_Tv=del_Tv,
                                            pump_ls=self.modes['pump_ls'],
                                            chi_rs_mode=self.modes['chi_rs'],
                                            convol_mode=self.modes['convol'],
                                            doppler_bd=self.modes['Doppler'],
                                            chem_eq=self.modes['chem_eq'])

        # convolute the slit function with the CARS spectrum
        I_as = np.convolve(I_as, nu_slit, 'same')

        # downsampling to the experimental spectral grid
        I_as_down = downsample(nu_expt, nu_f, I_as,
                               mode=self.modes['downsample'])**(
                                   0.5**self.modes['power_factor'])
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
        if self.modes['fit'] == 'room_fit':
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

        if self.modes['fit'] == 'T_fit':
            if path_room_fit is None:
                raise ValueError('Please provide path to (room_fit).pkl')
            if self.modes['chem_eq']:
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

        params = fit_model.make_params()
        params.add_many(*initi_params)
        if add_params is not None:
            params.add_many(*add_params)
        self.fit_result = fit_model.fit(np.nan_to_num(self.spec_cars), params,
                                        nu_expt=self.nu)

        if show_fit:
            report_fit(self.fit_result, show_correl=True, modelpars=params)
            self.fit_result.plot()
            # plt.show()
        # else:
        #     print('Best-fit T=%.2f +/- %.2f' % (
        #         self.fit_result.params['temperature'].value,
        #         self.fit_result.params['temperature'].stderr))

    def save_fit(self, dir_save, file_name='room_fit'):
        """
        Save the fitting results of room temperature CARS spectrum in
        a dictionary.

        Parameters
        ----------
        dir_save : str
            Path for saving the data.
        file_name : 'room_fit', str, optional
            Name of the file to be saved.
        """
        path_write = Path(dir_save) / (file_name + '.pkl')
        pkl_dump(path_write, self.fit_result)
