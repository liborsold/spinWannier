.. spinWannier documentation master file, created by
   sphinx-quickstart on Mon Sep 23 17:59:34 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |br| raw:: html

   <br />

=======================================
Welcome to spinWannier's documentation!
=======================================

The `spinWannier project on GitHub <https://github.com/liborsold/spinWannier/>`_ is a Python package for handling Wannier models with a spin operator, calculating the Wannier model quality and spin-texture plotting, as illustrated below.

.. CrXY example image
.. image::
   ./_static/images/spinWannier_example_figure.jpg
   :width: 800px
   :align: center

|br|
If you find this package useful, please cite `L. Vojáček*, J. M. Dueñas* et al., Nano Letters (2024) <https://pubs.acs.org/doi/10.1021/acs.nanolett.4c03029>`_.

Quickstart
================

.. code-block:: python
   
   pip install spinWannier

then 

.. code-block:: python
   
   git clone https://github.com/liborsold/spinWannier.git
   
and execute the ``./examples/spinWannier_use_example.ipynb`` Jupyter notebook to see an example of calculating the Wannier (spin) interpolation, 1D and 2D spin-texture plotting and Wannier model quality calculation.

Navigation
================
.. toctree::
   :maxdepth: 5

   theory
   example
   modules
   classes
   functions


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
