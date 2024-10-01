# spinWannier
**<i>Tools for handling Wannier models with a spin operator, calculating the Wannier model quality and spin-texture plotting.</i>** See the [documentation](https://liborsold.github.io/spinWannier/).

```
pip install spinWannier
```

If you find this package useful, please cite [L. Vojáček*, J. M. Dueñas* _et al._, Nano Letters (2024)](https://pubs.acs.org/doi/10.1021/acs.nanolett.4c03029). The underlying physics is summarized in Sec. S1.1 of the paper's [Supplementary](https://pubs.acs.org/doi/suppl/10.1021/acs.nanolett.4c03029/suppl_file/nl4c03029_si_001.pdf).

Example plot of Fermi-surface spin textures of CrXY (X,Y=S,Se,Te) monolayers:
<center><img src="https://github.com/user-attachments/assets/5204849c-0fa1-419f-9955-6c55c014babe" alt="spin_texture_example" width="550" /></center>

## Usage 
An example of use is given in ``./examples/spinWannier_use_example.ipynb``. It uses input files of a CrSeTe monolayer given in ``./examples/CrSeTe/``.

### 1. Load the model from ``wannier90`` files
```python
from spinWannier.WannierTBmodel import WannierTBmodel
model = WannierTBmodel(sc_dir='./sc', nsc_dir='./nsc', wann_dir='./wann', bands_dir='./bands')
```

### 2. Interpolate bands and spin along a high-symmetry path
```python
kpoint_matrix=[[(0.33,0.33,0.00), (0.00,0.00,0.00)],
               [(0.00,0.00,0.00), (0.50,0.00,0.00)],
               [(0.50,0.00,0.00), (0.33,0.33,0.00)]]

model.interpolate_bands_and_spin(kpoint_matrix, kpath_ticks=['K','G','M','K'], kmesh_2D=False)
model.plot1D_bands(yaxis_lim=[-6.6, 7.5], savefig=True, showfig=True)
```

<center><img src="https://github.com/user-attachments/assets/0172a2a3-450a-4a39-b223-2c629f1259e1" alt="spin_projected_bands_example" width="950" /></center>

### 3. Interpolate bands and spin on a 2D Brillouin zone
```python
model.interpolate_bands_and_spin(kpoint_matrix, kmesh_2D=True)
model.plot2D_spin_texture()
```
<center><img src="https://github.com/user-attachments/assets/f6b3f554-6801-4650-85e1-bb09d679b94b" alt="2D_spin_textures_example" width="950" /></center>

(In-plane spin projection as arrows, out-of-plane spin color-coded.)
<center><img src="https://github.com/user-attachments/assets/a336f039-1b9c-401d-a8d3-06e22ad259d8" alt="2D_spin_textures_all-in-one_example" width="400" /></center>


### 4. Calculate the error of Wannier interpolation
```python
model.wannier_quality(yaxis_lim=[-6.5, 7.5], savefig=True, showfig=True)
```
<center><img src="https://github.com/user-attachments/assets/d36a58e1-f9a1-4c1b-aab3-329f5c537378" alt="spin_projected_bands_example" width="950" /></center>

(The same information is also plotted as a function of energy, _integrated over k-space_.)
<center><img src="https://github.com/user-attachments/assets/ad971762-005e-40a5-ba48-9d9504e77d69" alt="spin_projected_bands_example" width="600" /></center>

(Spin magnitudes, _integrated over k-space_.)
<center><img src="https://github.com/user-attachments/assets/7200a663-1d5a-4dc8-a504-70a509115194" alt="spin_magnitudes" width="400" /></center>

(Their histogram, with most values close to 1, as expected.)
<center><img src="https://github.com/user-attachments/assets/3ee421ca-689c-4dae-b443-4707668fc9c6" alt="spin_magnitudes_histogram" width="400" /></center>





