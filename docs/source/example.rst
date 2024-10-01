Example
=======================

An example of use is given in ``./examples/spinWannier_use_example.ipynb``. It uses input files of a CrSeTe monolayer given in ``./examples/CrSeTe/``.

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
   https://github.com/user-attachments/assets/0172a2a3-450a-4a39-b223-2c629f1259e1
   :width: 950px
   :align: center

3. Interpolate bands and spin on a 2D Brillouin zone
--------------------------------------------------------
.. code-block:: python

    model.interpolate_bands_and_spin(kpoint_matrix, kmesh_2D=True)
    model.plot2D_spin_texture()

.. image::
   https://github.com/user-attachments/assets/f6b3f554-6801-4650-85e1-bb09d679b94b
   :width: 950px
   :align: center

(In-plane spin projection as arrows, out-of-plane spin color-coded.)

.. image::
   https://github.com/user-attachments/assets/a336f039-1b9c-401d-a8d3-06e22ad259d8
   :width: 400px
   :align: center

4. Calculate the error of Wannier interpolation
--------------------------------------------------------
.. code-block:: python

    model.wannier_quality(yaxis_lim=[-6.5, 7.5], savefig=True, showfig=True)

.. image::
   https://github.com/user-attachments/assets/d36a58e1-f9a1-4c1b-aab3-329f5c537378
   :width: 950px
   :align: center

(The same information is also plotted as a function of energy, _integrated over k-space.)

.. image::
   https://github.com/user-attachments/assets/ad971762-005e-40a5-ba48-9d9504e77d69
   :width: 550px
   :align: center

(Spin magnitudes, _integrated over k-space.)

.. image::
   https://github.com/user-attachments/assets/7200a663-1d5a-4dc8-a504-70a509115194
   :width: 350px
   :align: center

(Their histogram, with most values close to 1, as expected.)

.. image::
   https://github.com/user-attachments/assets/3ee421ca-689c-4dae-b443-4707668fc9c6
   :width: 350px
   :align: center
