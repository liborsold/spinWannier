from spinWannier.WannierTBmodel import WannierTBmodel

# Create a WannierTBmodel object
model = WannierTBmodel()
model.interpolate_bands_and_spin(kmesh_2D=False, kmesh_density=101, kmesh_2D_limits=[-0.5, 0.5], \
                                   save_bands_spin_texture=True)
model.plot1D_bands(fout='spin_texture_1D_home_made.jpg', yaxis_lim=[-8, 6])