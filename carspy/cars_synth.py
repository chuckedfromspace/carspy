"""Synthesize coherent anti-stokes Raman spectra for common gas species."""
import numpy as np
from .line_strength import LineStrength, linewidth_isolated
from ._constants import univ_const, chi_const
from .convol_fcn import gaussian_line, lorentz_line
from .utils import comp_normalize, eq_comp


class CarsSpectrum():
    """Basic class for synthesizing CARS spectrum."""

    def __init__(self, pressure=1, init_comp=None, chi_set="SET 1", **kwargs):
        """Input sample volume conditions.

        Parameters
        ----------
        pressure : float, optional
            Pressure in bars, by default 1.
        init_comp : dict, optional
            Initial composition in the measurement volume (prior to chemical
            reaction). If not given standard air composition is assumed as:

            .. code-block:: python

                {
                'N2': 0.79,
                'Ar': 0.0,
                'CO2': 0,
                'CO': 0,
                'H2': 0,
                'O2': 0.21,
                'H2O': 0
                }
        chi_set : str, optional
            Choose from the available set of non-resonant susceptibilities, by
            default 'SET 1':

            * 'SET 1': based on CARSFT :cite:`Palmer:89`.
            * 'SET 2': based on NRC :cite:`Parameswaran:88`.
            * 'SET 3': based on Eckbreth :cite:`Eckbreth:96`.

        Other Parameters
        ----------------
        species : str, optional
            Specify the molecule, by default 'N2'. Currently only supports
            'N2'.
        custom_dict : dict, optional
            Specify custom molecular constants, by default None. This can be
            used to modify default molecular constants and/or add custom
            species. The dictionary should contain all the necessary keys as in
            the default:

            .. code-block:: python

                ['we', 'wx', 'wy', 'wz', 'Be', 'De', 'alpha_e', 'beta_e',
                'gamma_e', 'H0', 'He', 'mu', 'MW', 'Const_Raman', 'G/A']
        """
        self.pressure = pressure

        if init_comp is None:
            # Assume air
            self.init_comp = {
                'N2': 0.79,
                'Ar': 0.0,
                'CO2': 0,
                'CO': 0,
                'H2': 0,
                'O2': 0.21,
                'H2O': 0,
                }
        else:
            # normalize the mole fraction to 1
            self.init_comp = comp_normalize(init_comp)

        # initiate line strength factors
        self.ls_factors = LineStrength(**kwargs)

        # load Universal constants
        self.Const_N = univ_const("Const_N")  # *1e22 [molecules/cm^3]
        self.C = univ_const("c")  # speed of light

        # load third order nonresonant susceptibilities
        self.chi_nrs = chi_const(chi_set)["SPECIES"]
        self.chi_nrs_T0 = chi_const(chi_set)["T0"]
        self.chi_nrs_P0 = chi_const(chi_set)["P0"]

        # load species-specific constants
        # alpha'/(2*pi*c*v0*m_reduced)^0.5 [*1e-11]
        self.Const_Raman = self.ls_factors.mc_dict['Const_Raman']
        # gamma'/alpha'
        self.pol_ratio = self.ls_factors.mc_dict['G/A']

    def num_dens(self, temperature):
        r"""Calculate number density in the probe volume.

        Parameters
        ----------
        temperature : float
            Temperature in [K].

        Returns
        -------
        float
            Number density in the probe volume in
            [:math:`\mathrm{molecules}/\mathrm{cm}^3`].
        """
        return self.pressure/temperature*self.Const_N

    def chi_nrs_est(self, temperature, local_comp=None):
        """Calculate effective local nonresonant susceptibility.

        Parameters
        ----------
        temperature : float
            Temperature in [K].
        local_comp : dict, optional
            A dictionary of local gas composition (percentile) for the purpose
            of calculating nonresonant background and collisional narrowing, by
            default None. If not given, initial composition will be used.
        """
        if local_comp is None:
            local_comp = self.init_comp
        local_comp = comp_normalize(local_comp)

        # Calculate the effective nonresonant susceptibility based on given
        # composition or air
        _common_keys = [_key for _key in self.chi_nrs if _key in local_comp]
        chi_nrs_eff = np.sum([self.chi_nrs[_key]*local_comp[_key]
                              for _key in _common_keys])

        # Convert to actual pressure and temperature condition
        chi_nrs_eff = chi_nrs_eff*(
            self.pressure/self.chi_nrs_P0*self.chi_nrs_T0/temperature)

        return chi_nrs_eff

    def trans_amp(self, v, j, branch=0):
        """Transition amplitude d or polarizability matrix element.

        It's related to Raman differential scattering cross-section and number
        density. Note that it is molecule based (divided number density N).

        Parameters
        ----------
        v : int
            Vibrational quantum number.
        j : int
            Rotational quantum number.
        branch : int, optional
            Q, S or O-branch with a value of 0, 2 or -2, by default 0.

        Returns
        -------
        d_squared : float
            Transition amplitude squared based on specified v, j and branch.
        """
        # Calculate the line strength factors using the LineStrength class
        pt_coeff, cd_coeff = self.ls_factors.int_corr(j, branch)

        # Calculate the branch-dependent d^2
        if branch in (2, -2):
            d_squared = (self.Const_Raman**2*(4/45)*pt_coeff
                         * self.pol_ratio**2*(v+1)*cd_coeff)
        elif branch == 0:
            d_squared = self.Const_Raman**2*(
                1 + (4/45)*pt_coeff*self.pol_ratio**2)*(v+1)*cd_coeff
        else:
            raise ValueError("Branch can only be 0, 2 or -2")

        return d_squared

    def relax_rate(self, temperature, j_i, j_j, fit_param=None):
        """Calculate relaxation rates.

        This is carried out using the modified exponential gap law
        (MEG) :cite:`Rahn:86`.

        .. note::
            Only considers Q-branch. O- and S-branches are taken as the same as
            Q-branch. It is also independent of vibrational state. Only valid
            for N2 at the moment.

        .. attention::

            Due to nuclear-spin selection rules for N2, gamma_ji is zero when
            (j_j-j_i) is odd. This would need to be adjusted for other types of
            molecules.

        Parameters
        ----------
        temperature : float
            Temperature in [K].
        j_i : int
            Rotational quantum number of the initial state.
        j_j : int
            Rotational quantum number of the final state.
        fit_param : list
            The four fitting parameters in a list: alpha, beta, sigma, m,
            default None. Needs to be provided.

        Returns
        -------
        gamma_ji, gamma_ij : float
            Upward and downward relaxation rate.
        """
        # Energies of the rovibrational states involved.
        Ej_i = self.ls_factors.term_values(0, j_i, 'Fv')
        del_E = self.ls_factors.term_values(0, j_j, 'Fv') - Ej_i

        # For a transition from j_i to j_j with i<j:
        if abs(j_i-j_j) % 2 != 0:
            # For N2 gamma_ji is zero when del_j is odd due to nuclear-spin
            # selection rules.
            gamma_ji = 0
            gamma_ij = 0
        else:
            alpha, beta, sigma, m = fit_param

            _term_1 = (1-np.exp(-m))/(
                1-np.exp(-m*temperature/295))*(295/temperature)**0.5
            _term_2 = ((1+1.5*1.44*Ej_i/temperature/sigma)/(
                1+1.5*1.44*Ej_i/temperature))**2

            gamma_ji = alpha*self.pressure/1.01325*_term_1*_term_2*np.exp(
                -beta*del_E*1.44/temperature)
            gamma_ij = gamma_ji*(2*j_i+1)/(2*j_j+1)*np.exp(
                del_E*1.44/temperature)

        return gamma_ji, gamma_ij

    def relax_mat(self, temperature, js=70, mode='MEG'):
        """Calculate relaxation rate matrix.

        The matrix elements are calculated using different exponential gap
        laws. The calculations are conducted for a fixed number of j levels,
        independent of v or branch.

        .. note::

            The (absolute) diagonal values are the HWHM of Raman linewidth.
            They are cross checked against NRC report :cite:`Parameswaran:88`
            on D4-D9 at 1 atm and 1550 K.

        Parameters
        ----------
        temperature : float
            Temperature in [K].
        js : int, optional
            Total number of rotational levels to be considered, by default 70.
        mode : str, optional
            By default 'MEG' is used. Possible options are:

            * 'MEG': Modified exponential gap law considering only N2-N2
              collisions :cite:`Rahn:86`. The fitting parameters are taken from
              :cite:`Woyde:90`.
            * 'EMEG': Extended MEG which weighs the contributions of N2, CO2
              and H2O :cite:`Woyde:90` (to be implemented).
            * 'XMEG': Extended MEG plus vibrational dephasing :cite:`Porter:90`
              (to be implemented).

        Returns
        -------
        2d array with the shape of (js, js)
            Relaxation matrix with the shape of js x js.
        """
        if mode != 'MEG':
            raise ValueError('Only MEG available at the moment!')
        # Define fitting parameters for N2-N2, N2-CO2, N2-H20
        #               alpha     beta, sigma, m
        #             [cm^-1/atm]
        fit_param_N2 = [0.0231, 1.67, 1.21, 0.1487]

        # Construct the matrix
        gamma_mat = np.zeros([js, js])
        for _i in range(js):
            for _j in range(_i+1, js):  # i.e., _i<_j
                gamma_mat[_j, _i], gamma_mat[_i, _j] = self.relax_rate(
                    temperature, _i, _j, fit_param_N2)

        # Calculate the diagonal
        for _i in range(js):
            gamma_mat[_i, _i] = -np.sum(gamma_mat[:, _i])

        return gamma_mat

    def peak_check(self, x_mol, temperature, v, j, branch=0, Gamma_j=None):
        r"""Calculate the theoretical peak intensity.

        This is done for the specified transition, assuming isolated lines.

        .. note::

            The purpose of this method is to check against the values tabulated
            in the NRC report :cite:`Parameswaran:88` on D4-D9. The agreement
            is excellent if x_mol is set to 1 and if the Gamma_j from NRC is
            used.

        Parameters
        ----------
        x_mol : float
            Mole fraction of probed molecule within [0, 1].
        temperature : float
            Temperature in the probe volume.
        v : int
            Vibrational quantum number.
        j : int
            Rotational quantum number.
        branch : int, optional
            Q, S or O-branch with a value of 0, 2 or -2, by default 0.
        Gamma_j : float, optional
            Raman linewidth, by default None. When not given, the value will be
            calculated from
            :mod:`carspy.line_strength.linewidth_isolated`.

        Returns
        -------
        float
            Peak transition amplitude based on specified v, j and branch.
        """
        # Species number density
        N = x_mol*self.num_dens(temperature)

        # Raman linewidth (depreciated, use MEG instead)
        if Gamma_j is None:
            Gamma_j = linewidth_isolated(self.pressure, temperature, j)

        # Calculate intensity
        del_pop = self.ls_factors.pop_factor(temperature, v, j, branch)
        d_squared = self.trans_amp(v, j, branch)

        return N*del_pop*d_squared/Gamma_j/(2*np.pi*self.C)

    def chi_rs_gmat(self, nu_s, temperature, vs=3, js=70, branches=(0,),
                    del_Tv=0.):
        """Resonant susceptibility based on G-matrix.

        This is implemented following :cite:`Koszykowski:85`.

        Parameters
        ----------
        nu_s : 1d array of floats
            Stokes frequencies (i.e., the calculation spectral domain).
        temperature : float
            Temperature in [K].
        vs : int, optional
            Total number of vibrational levels to be considered, by default 3.
        js : int, optional
            Total number of rotational levels to be considered, by default 70.
        branches : list of int, optional
            Branches to be considered, by default (0,) (i.e., only Q-branch).
        del_Tv : float, optional
            Absolute difference between vibrational and rotational temperature,
            by default 0.

        Returns
        -------
        1d array (complex)
            Theoretical resonant contributions (complex) based on G-matrix.
        """
        # Construct the v-branch-independent relaxation rate matrix
        gamma_mat = self.relax_mat(temperature, js)

        # For different v-branch combinations
        _js = np.arange(js)
        chi_rs = np.zeros_like(nu_s, dtype='complex128')
        for _branch in branches:
            for _v in np.arange(vs):
                # Calculate line positions
                nu_raman = self.ls_factors.line_pos(_v, _js, branch=_branch)

                # Construct the K_mat
                K_mat = np.diag(nu_raman) + gamma_mat*1j

                # Solve eigenvalue problem of K_mat
                eigvals, eigvec = np.linalg.eig(K_mat)
                eigvec_inv = np.linalg.inv(eigvec)

                # Compute the resonant intensity
                del_pop = self.ls_factors.pop_factor(temperature, _v, _js,
                                                     branch=_branch,
                                                     del_Tv=del_Tv)
                d = (self.trans_amp(_v, _js, branch=_branch))**0.5
                _term_l = d @ eigvec
                _term_r = eigvec_inv @ np.diag(del_pop) @ d
                _term = _term_l*_term_r

                for _j in _js:
                    _term_b = ((-nu_s + np.real(eigvals[_j]))**2
                               + np.imag(eigvals[_j])**2)
                    # A 1/2 factor is necessary to match the magnitude from
                    # isolated line assumption
                    chi_rs += 1/2*_term[_j]*np.conj(
                        -nu_s + eigvals[_j])/_term_b

        # A factor of c [cm/s] needs to be considered to convert cm^-1 to s^-1
        # by 2*pi*c
        return chi_rs/2/np.pi/self.C

    def chi_rs_isolated(self, nu_s, temperature, vs=3, js=70,
                        branches=(0, 2, -2), del_Tv=0):
        """Resonant susceptibility based on isolated line approximation.

        Parameters
        ----------
        nu_s : 1d array of floats
            Stokes frequencies (i.e., the calculation spectral domain).
        temperature : float
            Temperature in the probe volume.
        vs : int, optional
            Total number of vibrational levels to be considered, by default 3.
        js : int, optional
            Total number of rotational levels to be considered, by default 70.
        branches : list of int, optional
            Branches to be considered, by default (2, -2, 0).
        del_Tv : float, optional
            Absolute difference between vibrational and rotational temperature,
            by default 0.

        Returns
        -------
        1d array, 1d array
            Theoretical resonant contributions (complex) based on isolated line
            assumption.
        """
        # Construct the v-branch-independent relaxation rate matrix
        gamma_mat = self.relax_mat(temperature, js)
        Gamma_js = 2*abs(np.diag(gamma_mat))

        # Sum over all the transitions considered
        _js = np.arange(js)
        chi_rs = np.zeros_like(nu_s, dtype='complex128')
        for _branch in branches:
            for _v in np.arange(vs):
                # Calculate line positions, population factors, Gamma_j for all
                # js:
                nu_raman = self.ls_factors.line_pos(_v, _js, branch=_branch)
                del_pop = self.ls_factors.pop_factor(temperature, _v, _js,
                                                     branch=_branch,
                                                     del_Tv=del_Tv)
                d_squared = self.trans_amp(_v, _js, branch=_branch)

                # Calculate resonant contribution from each transition
                for _j in _js:
                    del_nu = nu_raman[_j] - nu_s
                    chi_rs += del_pop[_j]*d_squared[_j]*(
                        2*del_nu + Gamma_js[_j]*1j)/(
                            4*del_nu**2 + Gamma_js[_j]**2)

        # A factor of c [cm/s] needs to be considered to convert cm^-1 to s^-1
        # by 2*pi*c
        return chi_rs/2/np.pi/self.C

    def signal_as(self, temperature, x_mol=None, nu_s=None, pump_lw=None,
                  synth_mode=None, eq_func=eq_comp, **kwargs):
        r"""Anti-Stokes signal convoluted with a chosen laser lineshape.

        .. note::

            The Kataoka (or T-K) convolution :cite:`Kataoka:82, Teets:84` is a
            simplified form derived by :cite:`Farrow:85`. It is essentially
            an average of convolve(chi, pump)**2 and convolve(chi**2, pump),
            which takes care of the cross coherence effect. Excellent
            agreements are found when compared with Figs.13-15 in the NRC
            report :cite:`Parameswaran:88`.

        Parameters
        ----------
        temperature : float
            Temperature in the probe volume.
        x_mol : float, optional
            Mole fraction of probed molecule, by default None.
        nu_s : 1d array of float, optional
            Stokes frequencies (i.e., the calculation spectral domain), by
            default None.
        pump_lw : float, optional
            Pump laser linewdith (FWHM) in [:math:`\mathrm{cm}^{-1}`],
            by default None (i.e., no laser convolution is performed).
        synth_mode : dict, optional
            A dictionary containing the control parameters for creating the
            CARS spectrum, by default:

            pump_ls : 'Gaussian'
                Choose between pump laser lineshape between 'Gaussian' and
                'Lorentzian'.
            chi_rs : 'G-matrix'
                The method to compute resonant susceptibility: 'isolated' or
                'G-matrix' (collisional narrowing).
            convol : 'Kataoka' or 'K'
                Two ways to convolve the laser line with the resonant CARS
                susceptibilities:

                * 'Yuratich' or 'Y': One convolution, no cross coherence
                  effects accounted for :cite:`Yuratich:79`.
                * 'Kataoka' or 'K': Cross-coherence convolution, following
                  :cite:`Kataoka:82, Teets:84`.
                  This is necessary if pump linewidth is comparable to the
                  Raman linewidth (as is often the case at low T) and if
                  nonresonant contribution competes with resonant signal.
            doppler_effect : True
                Whether or not to take into account additional Doppler
                broadening.
            chem_eq : False
                Whether or not to assume chemical equilibrium. If True, an
                `eq_func` needs to be provided.
        eq_func : func, optional
                A function used to calculate local gas composition based on
                temperature and initial gas composition. Default is a simple
                template employing `cantera`.

        Other Parameters
        ----------------
        vs : int, optional
            Total number of vibrational levels to be considered, by default 3.
        js : int, optional
            Total number of rotational levels to be considered, by default 70.
        branches : list of int, optional
            Branches to be considered, by default (2, -2, 0).
        del_Tv : float, optional
            Absolute difference between vibrational and rotational temperature,
            by default 0.

        Returns
        -------
        1d array of floats, 1d array of floats
            Spectral locations in wavenumbers and theoretical CARS spectrum. If
            pump_lw is not specified, I_as needs to be multiplied by 1e-30 to
            retain its actual physical magnitude in
            [:math:`\mathrm{cm}^3/\mathrm{erg}`].
        """
        if synth_mode is None:
            self.synth_mode = {'pump_ls': 'Gaussian',
                               'chi_rs': 'G-matrix',
                               'convol': 'Kataoka',
                               'doppler_effect': False,
                               'chem_eq': False,
                               }
        else:
            self.synth_mode = synth_mode

        if x_mol is None:
            x_mol = self.init_comp[self.ls_factors.species]

        if self.synth_mode['chem_eq']:
            gas_comp = eq_func(temperature, self.pressure, self.init_comp)
            gas_comp = comp_normalize(gas_comp)
            N_species = gas_comp[
                self.ls_factors.species]*self.num_dens(temperature)
            chi_nrs = self.chi_nrs_est(temperature, local_comp=gas_comp)*1e-18
            self.local_comp = gas_comp
        else:
            # Species number density
            N_species = x_mol*self.num_dens(temperature)
            # Estimated nonresonant contribution based on given gas composition
            chi_nrs = self.chi_nrs_est(temperature)*1e-18

        # Specify the spectral domain
        if nu_s is None:
            # Ideally the grid should be much finer to correctly cover the
            # high-J transitions
            nu_s = np.linspace(2260, 2345, num=10000)

        # Compute the resonant contribution
        if self.synth_mode['chi_rs'] == 'isolated':
            chi_rs = self.chi_rs_isolated(nu_s, temperature, **kwargs)
        elif self.synth_mode['chi_rs'] == 'G-matrix':
            chi_rs = self.chi_rs_gmat(nu_s, temperature, **kwargs)
        else:
            raise ValueError("Unknown method. Only 'isolated' or 'G-matrix' "
                             "are available")

        # Compute the signal without pump laser convolution
        chi = (N_species*chi_rs + chi_nrs)*1e15

        # Consider Doppler broadening
        if self.synth_mode['doppler_effect']:
            _lw = self.ls_factors.doppler_lw(temperature)
            _ls = gaussian_line(nu_s, (nu_s[0]+nu_s[-1])/2, _lw)

            chi = np.convolve(chi, _ls, 'same')

        I_as = np.abs(chi)**2

        # Convolution with pump laser profile
        if pump_lw is not None:
            pump = []
            # Compute the pump laser line
            if self.synth_mode['pump_ls'] == 'Gaussian':
                pump = gaussian_line(nu_s, (nu_s[0]+nu_s[-1])/2, pump_lw)
            elif self.synth_mode['pump_ls'] == 'Lorentzian':
                pump = lorentz_line(nu_s, (nu_s[0]+nu_s[-1])/2, pump_lw)

            # Compute the convolution with Yuratich method (one convolution)
            if self.synth_mode['convol'] in ('Yuratich', 'Kataoka', 'Y', 'K'):
                I_as = np.convolve(I_as, pump, 'same')

            # Compute the convolution with Kataoka method (two convolutions)
            if self.synth_mode['convol'] in ('Kataoka', 'K'):
                chi_convol = np.convolve(chi, pump, 'same')
                # A correction factor is needed for the convolve(A)**2 v.s.
                # convolve(A**2)
                _corr = nu_s[1] - nu_s[0]
                I_as = 1/2*(I_as + _corr*np.abs(chi_convol)**2)

        return nu_s, I_as
