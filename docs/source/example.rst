Example
=======================

An example of use is given in ``./examples/spinWannier_use_example.ipynb``. It uses input files of a CrSTe monolayer given in ``./examples/CrSTe/``.

1. Load the model from ``wannier90`` files
--------------------------------------------------------

.. code-block:: python

    from spinWannier.WannierTBmodel import WannierTBmodel
    model = WannierTBmodel(sc_dir='./sc', nsc_dir='./nsc', wann_dir='./wann', bands_dir='./bands')


2. Interpolate bands and spin along a high-symmetry path
--------------------------------------------------------
.. code-block:: python

    kpoint_matrix=[[(0.33,0.33,0.00), (0.00,0.00,0.00)],
                [(0.00,0.00,0.00), (0.50,0.00,0.00)],
                [(0.50,0.00,0.00), (0.33,0.33,0.00)]]

    model.interpolate_bands_and_spin(kpoint_matrix, kpath_ticks=['K','G','M','K'], kmesh_2D=False)
    model.plot1D_bands(yaxis_lim=[-6.6, 7.5], savefig=True, showfig=True)

.. image::
   https://github.com/user-attachments/assets/b9012f49-3ba0-41b4-b3ba-26c5315982ee
   :width: 950px
   :align: center

3. Interpolate bands and spin on a 2D Brillouin zone
--------------------------------------------------------
.. code-block:: python

    model.interpolate_bands_and_spin(kpoint_matrix, kmesh_2D=True)
    model.plot2D_spin_texture()

.. image::
   https://github.com/user-attachments/assets/29b3987b-9b2b-4c3a-aacc-0f14681e5c89
   :width: 400px
   :align: center

4. Calculate the error of Wannier interpolation
--------------------------------------------------------
.. code-block:: python

    model.wannier_quality(yaxis_lim=[-6.5, 7.5], savefig=True, showfig=True)

.. image::
   https://github.com/user-attachments/assets/e7ce96ed-044a-4b7c-bdb0-e15146a24cee
   :width: 950px
   :align: center

(The same information is also plotted as a function of energy, _integrated over k-space_.)

.. image::
   https://github.com/user-attachments/assets/127f78fd-be19-4342-b679-a58cd11e945d
   :width: 600px
   :align: center
