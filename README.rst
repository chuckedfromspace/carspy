======
CARSpy
======

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
        :target: https://opensource.org/licenses/BSD-3-Clause

.. image:: https://img.shields.io/pypi/v/carspy.svg
        :target: https://pypi.python.org/pypi/carspy

.. image:: https://img.shields.io/pypi/pyversions/carspy.svg
        :target: https://pypi.python.org/pypi/carspy

.. image:: https://travis-ci.com/chuckedfromspace/carspy.svg
        :target: https://travis-ci.com/chuckedfromspace/carspy

.. image:: https://readthedocs.org/projects/carspy/badge/?version=latest
        :target: https://carspy.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Synthesizing and fitting coherent anti-Stokes Raman spectra in Python.

* `Documentation`_
* `Project homepage`_

.. _`Documentation`: https://carspy.readthedocs.io
.. _`Project homepage`: https://github.com/chuckedfromspace/carspy

Background
----------

Having no access to the source codes of any of the existing CARS programs, I started this project simply as a way to learn more about CARS.
I ended up spending a lot of time sifting through literatures from the past decades, trying to figure out what were done to analyze experimental CARS spectra and how they were implemented.
The latter proved rather challenging as specific implementations weren't always laid out in sufficient (mathematical) details to facilitate comprehension and replication (e.g., things as trivial as units for different constants weren't always made clear in some publications).

In an effort to put together a fully-functioning CARS fitting program, I thought it would perhaps benefit other CARS practitioners (especially the newcomers) if I open source my implementations.
I hope to also benefit from this transparency and openness to public scrutiny. Although the "draft" code (not available in this public repo) already lives up to my original purpose (least-square fit of experimental broadband CARS spectra),
it is most likely not error-free and has a lot of room left for improvement.
Therefore, I plan to rewrite the important modules (spectrum synthesis and least-square fit) and slowly bring all features (see below) up to date.  I am also looking forward to feedbacks and potential collaborators from the community.

.. note::
        Nitrogen is currently the only species implemented/tested in ``carspy``. Other common species will be added in the future (or can be readily introduced via customization).

Quick start
-----------

.. code-block:: console

    $ pip install carspy

See `installation guide`_ for alternative methods.

.. _`installation guide`: https://carspy.readthedocs.io/en/latest/installation.html

Features
--------

* CARSpy (stands for **C**\oherent **A**\nti-Stokes **R**\aman **S**\pectrosco\ **py**\):

.. image:: https://raw.githubusercontent.com/chuckedfromspace/carspy/main/assets/carspy_struct.png
        :width: 100%
        :align: center
        :alt: carspy structure

* The CARS model:

.. image:: https://raw.githubusercontent.com/chuckedfromspace/carspy/main/assets/cars_model.png
        :width: 100%
        :align: center
        :alt: cars model

.. note::
        * The default chemical equilibrium solver based on ``cantera`` can be replaced by custom functions.
        * Voigt profile is implemented via numerical convolution of a Gaussian profile with the Raman lines.
        * Extended exponential gap model is not yet implemented.

Highlights
----------

* Option to incorporate equilibrium composition using an external chemical equilibrium calculator (such as ``cantera``), such that temperature is the only fitting parameter for thermometry
* Vibrational and rotational nonequilibrium: vibrational temperature can be varied independently from rotational temperature

Comparisons with CARSFT
-----------------------

.. figure:: https://raw.githubusercontent.com/chuckedfromspace/carspy/main/assets/vs_CARSFT_01.jpeg
    :width: 70%
    :alt: vs_CARSFT_01
    :figclass: align-center

    Figure 1 Synthesized CARS spectra in N2 at 1 atm, 2400 K, with a pump linewidth of 0.5 cm-1, using Voigt lineshape and cross-coherence convolution.

.. figure:: https://raw.githubusercontent.com/chuckedfromspace/carspy/main/assets/vs_CARSFT_02.jpeg
    :width: 70%
    :alt: vs_CARSFT_02
    :figclass: align-center

    Figure 2 Synthesized CARS spectra in N2 at 10 atm, 2400 K, with a pump linewidth of 0.5 cm-1, using modified exponential gap law (MEG) and cross-coherence convolution.

.. caution::
        There seems to exist a number of compiled versions of CARSFT that have likely been modified (in a hardcoded way) to suit specific purposes (e.g., artificially inflated nonresonant background and/or Raman linewidth).

        The version used for the comparisons here was likely optimized for dual-pump CARS, such that several important settings (isolated line, single/double convolution, MEG, etc) don't behave consistently. Small tweaks during the configuration setup (e.g., modifiers) were necessary to create theoretically correct spectra in CARSFT.

Roadmap
-------

The above features currently present in the draft code will be gradually improved and included in the ``main`` branch. Here is a tentative plan:

* (Implemented) Module for synthesizing CARS spectra (optional with ``cantera``)
* (Short-term) Module for least-square fit (with ``lmfit``)
* (Mid-term) Multiprocessing
* (Mid-term) Docs
* (Mid-term) Tutorials
* (Long-term) Other common diatomic species
* (Long-term) Dualpump/Wide CARS

Citation
--------

Please consider citing this repository if you use ``carspy`` for your publications as:

.. code-block:: bib

    @misc{Yin2021,
      author = {Yin, Zhiyao},
      title = {CARSpy: Synthesizing and fitting coherent anti-Stokes Raman spectra in Python},
      year = {2021},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/chuckedfromspace/carspy}}
    }

Acknowledgement
---------------

* A copy of the NRC report (TR-GD-013_1989) was kindly provided by Dr. Gregory Smallwood and his colleagues at NRC,
  which has significantly eased the difficulty of understanding some of the key theories in synthesizing CARS spectra.

* This package was initially created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
