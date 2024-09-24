# spinWannier
**<i>Tools for handling Wannier models with a spin operator, calculating the Wannier model quality and spin-texture plotting.</i>** See the [documentation](https://liborsold.github.io/spinWannier/).

```
pip install spinWannier
```

If you find this package useful, please cite [L. Vojáček*, J. M. Dueñas* _et al._, Nano Letters (2024)](https://pubs.acs.org/doi/10.1021/acs.nanolett.4c03029). The underlying physics is summarized in Sec. S1.1 of the the paper's [Supplementary](https://pubs.acs.org/doi/suppl/10.1021/acs.nanolett.4c03029/suppl_file/nl4c03029_si_001.pdf).


## Usage 

An example of use is given in ``./examples/spinWannier_use_example.ipynb``. It uses input files of a CrTe<sub>2</sub> monolayer given in ``./examples/wannier_files_CrTe2/``.


### 2D Brillouin zone spin textures
Example plot of Fermi-surface spin textures of CrXY (X,Y=S,Se,Te) monolayers, 
as calculated in [L. Vojáček*, J. M. Dueñas* et al., Nano Letters (2024)](https://pubs.acs.org/doi/10.1021/acs.nanolett.4c03029):
<center><img src="https://github.com/user-attachments/assets/5204849c-0fa1-419f-9955-6c55c014babe" alt="spin_texture_example" width="550" /></center>

### 1D spin-projected band structure
Example of a spin-projected band structure plot by ``spinWannier`` in ``./examples/spinWannier_use_example.ipynb``:
<center><img src="https://github.com/user-attachments/assets/b9012f49-3ba0-41b4-b3ba-26c5315982ee" alt="spin_projected_bands_example" width="950" /></center>

### Error of Wannier interpolation
Produced by ``spinWannier`` in ``./examples/spinWannier_use_example.ipynb``:
<center><img src="https://github.com/user-attachments/assets/e7ce96ed-044a-4b7c-bdb0-e15146a24cee" alt="spin_projected_bands_example" width="950" /></center>

The same, but _integrated over k-space_, plotted as a function of energy:
<center><img src="https://github.com/user-attachments/assets/127f78fd-be19-4342-b679-a58cd11e945d" alt="spin_projected_bands_example" width="600" /></center>




