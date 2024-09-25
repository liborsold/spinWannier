# spinWannier
**<i>Tools for handling Wannier models with a spin operator, calculating the Wannier model quality and spin-texture plotting.</i>** See the [documentation](https://liborsold.github.io/spinWannier/).

```
pip install spinWannier
```

If you find this package useful, please cite [L. Vojáček*, J. M. Dueñas* _et al._, Nano Letters (2024)](https://pubs.acs.org/doi/10.1021/acs.nanolett.4c03029). The underlying physics is summarized in Sec. S1.1 of the paper's [Supplementary](https://pubs.acs.org/doi/suppl/10.1021/acs.nanolett.4c03029/suppl_file/nl4c03029_si_001.pdf).

Example plot of Fermi-surface spin textures of CrXY (X,Y=S,Se,Te) monolayers:
<center><img src="https://github.com/user-attachments/assets/5204849c-0fa1-419f-9955-6c55c014babe" alt="spin_texture_example" width="550" /></center>

## Usage 
An example of use is given in ``./examples/spinWannier_use_example.ipynb``. It uses input files of a CrSTe monolayer given in ``./examples/CrSTe/``.

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
<center><img src="https://github.com/user-attachments/assets/b9012f49-3ba0-41b4-b3ba-26c5315982ee" alt="spin_projected_bands_example" width="950" /></center>

### 3. Interpolate bands and spin on a 2D Brillouin zone
```python
model.interpolate_bands_and_spin(kpoint_matrix, kmesh_2D=True)
model.plot2D_spin_texture()
```
<center><img src="https://github.com/user-attachments/assets/29b3987b-9b2b-4c3a-aacc-0f14681e5c89" alt="2D_spin_textures_example" width="400" /></center>

### 4. Calculate the error of Wannier interpolation
```python
model.wannier_quality(yaxis_lim=[-6.5, 7.5], savefig=True, showfig=True)
```
<center><img src="https://github.com/user-attachments/assets/e7ce96ed-044a-4b7c-bdb0-e15146a24cee" alt="spin_projected_bands_example" width="950" /></center>

(The same information is also plotted as a function of energy, _integrated over k-space_.)
<center><img src="https://github.com/user-attachments/assets/127f78fd-be19-4342-b679-a58cd11e945d" alt="spin_projected_bands_example" width="600" /></center>




