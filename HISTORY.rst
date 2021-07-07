==========
Change Log
==========

0.6.1 (2021-07-07)
------------------

* Fix the discontinuity problem in the asymmetric Voigt profile

0.6.0 (2021-07-05)
------------------

* Add a peak-finding function for calibrating the spectrometer based on Ar or Hg spectra.
* Implement a fitting routine for the slit function based either on asymmetric Gaussian or Voigt profiles.

0.5.0 (2021-05-17)
------------------

* Add link to a standalone web application that demonstrates the basic functions for synthesizing CARS spectra

0.4.2 (2021-03-12)
------------------

* Add missing json data files to the package.

0.4.1 (2021-03-10)
------------------

* Implement an asymmetric "super" Voigt function for better fitting slit function.

0.4.0 (2021-03-03)
------------------

* Implement least-square fitting routine with ``lmfit`` (optional).
* Add documentations for the least-square fitting module.
* Add usage examples for both synthesizing and fitting CARS spectra.

0.3.0 (2021-02-25)
------------------

* Implement optional function for calculating local gas composition at chemical equilibrium using ``cantera``.
* Add documentations for secondary modules.
* Add bibtex style references in documentations.

0.2.1 (2021-02-14)
------------------

* Change Sphinx theme.
* Add existing modules to docs.
* Fix format errors in docstrings.

0.2.0 (2021-02-13)
------------------

* Implement modules for synthesizing N2 CARS spectra.

0.1.0 (2021-02-13)
------------------

* First release on PyPI.
