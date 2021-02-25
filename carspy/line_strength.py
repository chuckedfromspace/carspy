"""Calculate Raman linestrength for common gas species."""
import numpy as np
from ._constants import mol_const, univ_const


def linewidth_isolated(pressure, temperature, j):
    r"""Temperature-dependent Raman linewidths of N2 from :cite:`Farrow:82`.

    .. admonition:: Deprecated
       :class: attention

       Use the MEG to calculate rate matrix and isolated linewidth.

    Parameters
    ----------
    pressure : float
        Pressure in [bar].
    temperature : float
        Temperature in [K].
    j : int
        Rotational quantum number.

    Returns
    -------
    Gamma_j : float
        Raman linewidth (FWHM) in [:math:`\mathrm{cm}^{-1}`].
    """
    if temperature > 600:
        _Gamma = 2*(1.79534e-2 - 6.3087e-4*j + 4.7995e-5*j**2
                    - 1.5139e-6*j**3 + 1.50467e-8*j**4)
        Gamma = _Gamma*pressure/0.84*(1750/temperature)**0.5
    else:
        _Gamma = 2*(1.99e-2 - 4.575e-4*j)
        Gamma = _Gamma*pressure/0.34*(295/temperature)**0.5

    return Gamma


class LineStrength():
    """A collection of methods concerning line strength in CARS Spectrum."""

    def __init__(self, species='N2', custom_dict=None):
        """Input molecular constants.

        Parameters
        ----------
        species : str, optional
            Specify the molecule, by default 'N2'. Currently only
            supports 'N2'.
        custom_dict : dict, optional
            Specify custom molecular constants, by default None. This can be
            used to modify default molecular constants and/or add custom
            species. The dictionary should contain all the necessary keys as in
            the default:

            .. code-block:: python

                ['we', 'wx', 'wy', 'wz', 'Be', 'De', 'alpha_e', 'beta_e',
                'gamma_e', 'H0', 'He', 'mu', 'MW', 'Const_Raman', 'G/A']

        """
        self.species = species
        if custom_dict is None:
            if species != 'N2':
                raise ValueError('Only N2 available at the moment! Please '
                                 'provide custom molecular constants for a '
                                 'different species')
            self.mc_dict = mol_const(species)
        else:
            self.mc_dict = custom_dict

        # load universal constant
        self.Const_D = univ_const('Const_D')

    def int_corr(self, j, branch=0):
        r"""Intensity corrections Q, S and O-branches.

        * Placzek-Teller coefficient.
        * Centrifugal distortion caused by vibration-rotation interaction.

        .. warning::

            The Herman-Wallis factor for centrifugal distortion may need to be
            updated. It seems the commonly used James-Klemperer expression
            (as used in this code) for Q-branch is not correct. This matters
            more for lighter molecules like H2. See :cite:`Marrocco:07`
            for more details.

        .. note::

            It seems CARSFT :cite:`Palmer:89` has set cd_corr=1 for Q-branch
            transitions. Here the formula from NRC :cite:`Parameswaran:88`
            is taken. This causes a relative intensity
            change in Q- vs O/S-braches. Since O/S-branches are significantly
            weaker around v=0-1, 1-2 transitions for N2
            (~2330 :math:`\mathrm{cm}^{-1}`), this difference is unlikely to
            raise any issue.

        Parameters
        ----------
        j : int
            Rotational quantum number.
        branch : int, optional
            Q, S or O-branch with a value of 0, 2 or -2, by default 0.

        Returns
        -------
        pt_coeff,  cd_corr : float
            Placzek-Teller coefficient and centrifugal distortion correction
            for the specific j-branch combination.
        """
        mc_dict = self.mc_dict
        pt_coeff, cd_corr = 0, 1
        # Calculate the coefficients
        if branch == 0:  # Q branch
            pt_coeff = j*(j+1)/(2*j-1)/(2*j+3)
            cd_corr = 1-6*mc_dict['Be']**2/mc_dict['we']**2*j*(j+1)
        elif branch == -2:  # O branch
            pt_coeff = 3*j*(j-1)/2/(2*j+1)/(2*j-1)
            cd_corr = (1 + 4*mc_dict['Be']/mc_dict[
                'we']*mc_dict['mu']*(2*j-1))**2
        elif branch == 2:  # S branch
            pt_coeff = 3*(j+1)*(j+2)/2/(2*j+1)/(2*j+3)
            cd_corr = (1 - 4*mc_dict['Be']/mc_dict[
                'we']*mc_dict['mu']*(2*j+3))**2

        return pt_coeff, cd_corr

    def term_values(self, v, j, mode='sum'):
        """Calculate the term values for an anharmonic vibrating rotor.

        Parameters
        ----------
        v : int
            Vibrational quantum number.
        j : int
            Rotational quantum number.
        mode : str, optional
            Determine what to return, by default 'sum':

            * 'sum' returns the total term values Gv + Fv
            * 'Gv' returns only Gv
            * 'Fv' returns only Fv

        Returns
        -------
        float
            Term values for the specified ro-vibrational level depending on
            the mode selected.
        """
        mc_dict = self.mc_dict
        Gv, Fv = [], []
        if mode in ('sum', 'Fv'):
            # Rotational contribution
            Bv = (mc_dict['Be'] - mc_dict['alpha_e']*(v+0.5)
                  + mc_dict['gamma_e']*(v+0.5)**2)
            Dv = mc_dict['De'] + mc_dict['beta_e']*(v+0.5)
            Hv = mc_dict['H0'] + mc_dict['He']*(v+0.5)

            Fv = Bv*j*(j+1) - Dv*j**2*(j+1)**2 + Hv*j**3*(j+1)**3

        if mode in ('sum', 'Gv'):
            # Vibrational contribution
            Gv = (mc_dict['we']*(v+0.5) - mc_dict['wx']*(v+0.5)**2
                  + mc_dict['wy']*(v+0.5)**3 + mc_dict['wz']*(v+0.5)**4)

        output = []
        if mode == 'sum':
            output = Gv + Fv
        elif mode == 'Gv':
            output = Gv
        elif mode == 'Fv':
            output = Fv

        return output

    def line_pos(self, v, j, branch=0):
        """Line position of anti-Stokes transitions.

        .. note::

            Validated against information from NRC Report TR-GD-013 (B21 Tabel
            B-1).

        Parameters
        ----------
        v : int
            Vibrational quantum number
        j : int
            Rotational quantum number
        branch : 0, int, optional
            Q, S or O-branch with a value of 0, 2 or -2

        Returns
        -------
        float
            Line position in wavenumber cm^(-1) for the given species at
            specified ro-vibrational levels
        """
        return self.term_values(v+1, j+branch) - self.term_values(v, j)

    def pop_frac(self, temperature, v, j, del_Tv=0.0, vs=20, js=100):
        """Calculate ro-vibrational partition functions.

        Parameters
        ----------
        temperature : float
            Translational (equilibrium) temperature of the measurement volume
            in [K].
        v : int
            Vibrational quantum number.
        j : int
            Rotational quantum number.
        del_Tv : float
            The amount Tv exceeds Tr, by default 0.0.This is only used when
            stimulated Raman process competes with CARS and Tv becomes
            obviously higher than Tr.
        vs : int, optional
            Total number of vibrational levels to be considered, by default 20.
        js : int, optional
            Total number of rotational levels to be considered, by default 100.

        Returns
        -------
        fvj : float
            Population fraction of (v, j) state at the given temperature.
        """
        def Gj(j):
            """Degeneracy from nuclear spin.

            Currently availabe for 3 popular diatomic molecules:
                * N2: Even j levels have Gj=6; Odd j levels have Gj=3
                * H2: Even j levels have Gj=1; Odd j levels have Gj=3
                * O2: Even j levels have Gj=0; odd j levels have Gj=1
            """
            _Gj = 1
            if self.species == 'N2':
                _Gj = 3*(2+(-1)*(j % 2))
            elif self.species == 'O2':
                _Gj = j % 2
            elif self.species == 'H2':
                _Gj = 2*(j % 2) + 1

            return _Gj

        def rho_v(v):
            return np.exp(-1.44/(
                temperature + del_Tv)*self.term_values(v, 0, mode='Gv'))

        def rho_r(v, j):
            return (2*j + 1)*Gj(j)*np.exp(
                -1.44/temperature*self.term_values(v, j, mode='Fv'))

        # Vibrational partition function
        Qv = rho_v(np.arange(vs)).sum()

        # Rotational partition function (depends on v)
        Qr = rho_r(v, np.arange(js)).sum()

        # Population fraction
        fvj = 1/Qv/Qr*rho_v(v)*rho_r(v, j)

        return fvj

    def pop_factor(self, temperature, v, j, branch=0, **kwargs):
        """Calculate population factor in the line intensity weight factor.

        This is the population difference between the lower and upper states
        for the specified transition (v, j, branch) at the given temperature.

        .. note::

            Validated against information from NRC Report TR-GD-013 (D4).

        Parameters
        ----------
        temperature : float
            Translational (equilibrium) temperature of the measurement volume.
        v : int
            Vibrational quantum number.
        j : int or array of int
            Rotational quantum number. Recommended here to use an array if
            multiple j-levels need to be computed. This reduces the necessity
            of extra for-loops for j when synthesizing spectrum.
        branch : int, optional
            Q, S or O-branch with a value of 0, 2 or -2, by default 0.

        Other Parameters
        ----------------
        All keyword arguments from ``pop_frac``.

        Returns
        -------
        float
            Fractional population difference between the lower and upper
            states.
        """
        f_low = self.pop_frac(temperature, v, j, **kwargs)
        f_up = self.pop_frac(temperature, v+1, j+branch, **kwargs)

        return f_low - f_up

    def doppler_lw(self, temperature, nu_0=2300.):
        r"""Calculate the Doppler broadening FWHM of the transitions.

        .. warning::

            This simple implementation for Doppler broadening may not be
            accurate enough for Dualpump- or Wide-CARS.

        Parameters
        ----------
        temperature : float
            Translational (equilibrium) temperature of the measurement volume.
        nu_0 : float, optional
            Transition center frequency, by default 2300 in
            [:math:`\mathrm{cm}^{-1}`]. It should actually be caclulated with
            ``line_pos`` for each single rotational transition. But to speed up
            computation, a single value is assumed for the entire
            CARS spectrum.

        Returns
        -------
        float
            FWHM of the Doppler broadening
        """
        return nu_0*(temperature/self.mc_dict['MW'])**0.5*self.Const_D
