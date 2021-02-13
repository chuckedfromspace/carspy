import os
import json


def mol_const(species='N2'):
    """Load molecular constants

    Parameters
    ----------
    species : str, optional
        Target molecular species, by default 'N2'. Currently only 'N2' is
        available
    """
    fpath = os.path.join(os.path.dirname(__file__), "_MOL_CONST.json")
    with open(fpath, 'r') as f:
        _mol_const = json.load(f)[species]

    # extract only  the necessary ones
    _keys = ['we', 'wx', 'wy', 'wz', 'Be', 'De', 'alpha_e', 'beta_e',
             'gamma_e', 'H0', 'He', 'mu', 'MW', 'Const_Raman', 'G/A']
    _mol_const = {_key: _mol_const[_key]['value'] for _key in _keys}

    return _mol_const


def univ_const(dict_key='Const_N'):
    """Load universal constants

    Parameters
    ----------
    dict_key : str, optional
        Specify the item to load from default _constants/_UNIV_CONST.json, by
        default 'Const_N'
    """
    fpath = os.path.join(os.path.dirname(__file__), "_UNIV_CONST.json")
    with open(fpath, 'r') as f:
        _mol_const = json.load(f)

    return _mol_const[dict_key]['value']


def chi_const(use_set='SET 1'):
    """Load third order nonresonant susceptibilities

    Parameters
    ----------
    use_set : str, optional
        Choose from the available set of susceptibilities, by default 'SET 1':
        - SET 1: based on CARSFT
        - SET 2: based on NRC
        - SET 3: based on Eckbreth
    """
    fpath = os.path.join(os.path.dirname(__file__), "_CHI_NRS.json")
    with open(fpath, 'r') as f:
        _chi_const = json.load(f)

    return _chi_const[use_set]


def get_const(const_type='mol', show=False):
    """Get default constants stored under _constants/

    Parameters
    ----------
    const_type : str, optional
        Type of constants to print. Choose from 'mol', 'univ' and 'chi',
        by default 'mol'
    """
    fname = []
    if const_type == 'mol':
        fname = "_MOL_CONST.json"
    elif const_type == 'univ':
        fname = "_UNIV_CONST.json"
    elif const_type == 'chi':
        fname = "_CHI_NRS.json"

    fpath = os.path.join(os.path.dirname(__file__), fname)
    with open(fpath, 'r') as f:
        _const = json.load(f)

    if show:
        print(json.dumps(_const, indent=4))
    else:
        return _const
