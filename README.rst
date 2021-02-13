.. |ss| raw:: html

    <strike>

.. |se| raw:: html

    </strike>

======
CARSpy
======

.. image:: https://img.shields.io/pypi/v/carspy.svg
        :target: https://pypi.python.org/pypi/carspy

.. image:: https://img.shields.io/travis/chuckedfromspace/carspy.svg
        :target: https://travis-ci.com/chuckedfromspace/carspy

.. image:: https://readthedocs.org/projects/carspy/badge/?version=latest
        :target: https://carspy.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Synthesizing and fitting coherent anti-Stokes Raman spectra (CARS) in Python

* Free software: BSD-3-Clause license
* Documentation: https://carspy.readthedocs.io (WIP).

Background
----------

Having no access to the source codes of any of the existing CARS programs, I started this project simply as a way to learn more about CARS.
I ended up spending a lot of time sifting through literatures from the past decades, trying to figure out what were done to analyze experimental CARS spectra and how they were implemented.
The latter proved rather challenging as specific implementations weren't always laid out in sufficient (mathematical) details to facilitate comprehension and replication (e.g., things as trivial as units for different constants weren't always made clear in some publications).

In an effort to put together a fully-functioning CARS fitting program, I thought it would perhaps benefit other CARS practitioners (especially the newcomers) if I open source my implementations.
I hope to also benefit from this transparency and openness to public scrutiny. Although the "draft" code (not available in this public repo) already lives up to my original purpose (least-square fit of experimental broadband CARS spectra),
it is most likely not error-free and has a lot of room left for improvement.
Therefore, I plan to rewrite the important modules (spectrum synthesis and least-square fit) and slowly bring all features (see below) up to date.  I am also looking forward to feedbacks and potential collaborators from the community.

**NOTE**: Nitrogen is currently the only species implemented/tested in ``carspy``. Other common species will be added in the future (or can be readily introduced via customization).

Features
--------

* CARSpy:

.. image:: https://github.com/chuckedfromspace/carspy/blob/main/assets/carspy_struct.png
    :width: 100%
    :align: center
    :alt: carspy structure

* The CARS model:

.. image:: https://github.com/chuckedfromspace/carspy/blob/main/assets/cars_model.png
    :width: 100%
    :align: center
    :alt: cars model

Highlights
----------

1. Option to incorporate equilibrium composition using an external chemical equilibrium calculator (such as ``cantera``), such that temperature is the only fitting parameter for thermometry
2. Vibrational and rotational nonequilibrium: vibrational temperature can be varied independently from rotational temperature

Comparisons with CARSFT
-----------------------

.. figure:: https://github.com/chuckedfromspace/carspy/blob/main/assets/vs_CARSFT_01.jpeg
    :width: 50%
    :align: center
    :alt: vs_CARSFT_01
    :figclass: align-center

    Figure 1 Synthesized CARS spectra in N2 at 1 atm, 2400 K, with a pump linewidth of 0.5 cm-1, using Voigt lineshape and cross-coherence convolution.

.. figure:: https://github.com/chuckedfromspace/carspy/blob/main/assets/vs_CARSFT_02.jpeg
    :width: 50%
    :align: center
    :alt: vs_CARSFT_02
    :figclass: align-center

    Figure 2 Synthesized CARS spectra in N2 at 10 atm, 2400 K, with a pump linewidth of 0.5 cm-1, using modified exponential gap law (MEG) and cross-coherence convolution.

Roadmap
-------

The above features currently present in the draft code will be gradually improved and included in the ``main`` branch. Here is a tentative plan:

1. |SS| Module for synthesizing CARS spectra |SE|
2. Module for least-square fit (with ``lmfit``)
3. Multiprocessing
4. Docs
5. Tutorials
6. Other common diatomic species
7. Dualpump/Wide CARS

Citation
--------

Please consider citing this repository if you use carspy for your research as:

.. code-block:: bib

    @misc{Yin2021,
      author = {Yin, Zhiyao},
      title = {CARSpy: Synthesizing and fitting coherent anti-Stokes Raman spectra (CARS) in Python},
      year = {2021},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/chuckedfromspace/carspy}}
    }
