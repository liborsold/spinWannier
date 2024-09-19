"""Fit a generalized Rashba model to a bandstructure, either DFT or Wannier.

Returns:
    _type_: _description_
"""

from mimetypes import init
from multiprocessing.sharedctypes import Value
import numpy as np
from numpy import cos, sin, arctan2
import pickle
from scipy.optimize import minimize
from plot_2D_spin_texture import fermi_surface_spin_texture
import sys
sys.path.append('/home/lv268562/bin/python_scripts/wannier_to_tight_binding')
from wannier_utils import load_lattice_vectors, reciprocal_lattice_vectors, get_kpoint_path, get_2D_kpoint_mesh, check_file_exists, \
                          outer
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.markers import MarkerStyle
import os
from copy import copy

n_bands = 4

# -------- CONSTANTS ---------
  # Pauli matrices
sigma_0  = np.eye(2)
sigma_x  = np.array([[0, 1], [1, 0]])
sigma_y  = np.array([[0, -1j], [1j, 0]])
sigma_z  = np.array([[1, 0], [0, -1]])

# Pauli matrices (for n_bands with spin as the main blocks)
sigma_0_n = np.eye(n_bands)
sigma_x_n = np.zeros((n_bands, n_bands))
sigma_x_n[np.eye(n_bands, k=n_bands//2, dtype='bool')] = 1
sigma_x_n[np.eye(n_bands, k=-n_bands//2, dtype='bool')] = 1
sigma_y_n = np.zeros((n_bands, n_bands), dtype=np.complex64)
sigma_y_n[np.eye(n_bands, k=n_bands//2, dtype='bool')] = -1j
sigma_y_n[np.eye(n_bands, k=-n_bands//2, dtype='bool')] = 1j
sigma_z_n = np.diag([1]*(n_bands//2) + [-1]*(n_bands//2))
C_kinetic = 7.6211      # E_kinetic (eV) = 1/(2*m_star) * C_kinetic * k^2 (1/Angstrom^2)

  # p-orbital linear combination matrices
rho_0 = np.eye(2)
rho_x = np.array([[0, 1], [1, 0]])      # equivalent to sigma_x
rho_y = np.array([[0, -1j], [1j, 0]])   # equivalent to sigma_y
rho_z = np.array([[1, 0], [0, -1]])     # equivalent to sigma_z

  # save combined matrices
  # create combined names
# a = ['s_0', 's_x', 's_y', 's_z']
# b = ['rho_0', 'rho_x', 'rho_y', 'rho_z']
# l = []
# for i in a:
#     ls = []
#     for j in b:
#         ls.append(f"{i}*{j}")
#     l.append(ls)
# repr(l)

sig_rho_names = [['s_0*rho_0', 's_0*rho_x', 's_0*rho_y', 's_0*rho_z'], ['s_x*rho_0', 's_x*rho_x', 's_x*rho_y', 's_x*rho_z'], ['s_y*rho_0', 's_y*rho_x', 's_y*rho_y', 's_y*rho_z'], ['s_z*rho_0', 's_z*rho_x', 's_z*rho_y', 's_z*rho_z']]
sig_rho = {}
for i, sigma in enumerate([sigma_0, sigma_x, sigma_y, sigma_z]):
    for j, rho in enumerate([rho_0, rho_x, rho_y, rho_z]):
        sig_rho[sig_rho_names[i][j]] = outer(sigma, rho)


class RashbaFitter():
    def __init__(self) -> None:
        """_summary_
        """

    def set_Fermi(self, E_F):
        """Set the Fermi level.

        Args:
            E_F (float): Fermi level in units of eV.
        """
        self.E_F = E_F

    def set_calc_2D(self, calc_2D):
        """Set calc_2D parameter, which tells if a 2D band structure is being fitted.

        Args:
            calc_2D (bool): is the data 2D band structure or not
        """
        self.calc_2D = calc_2D

    def save_hyperparameters(self, hyperparameters, hyperparameter_names):
        """Save hyperparameters of calculation for later saving and reference."""
        self.hyperparameters = hyperparameters
        self.hyperparameters_names = hyperparameter_names

    def load_spin_texture(self, path, band_numbers, shift_by_fermi=False, energy_lim=[-0.1, 2], selection_by_energy=True, metric='Euclidean', k_lim=1, kx_lims=[-1,1], ky_lims=[-1,1]):
        """Load DFT bands + spin texture information. Select the band numbers and k-space limits for fitting (you want to 
        avoid the bands crossing, i.e., becoming different bands.)

        Args:
            path (_type_): _description_
            band_numbers (int 2x1 array): band numbers to be fitted
            kx_lims (float 2x1 array): limits of kx, units of 1/Angstrom.
        """
        with open(path, 'br') as fr:
            data = pickle.load(fr)

        kpoints = np.array(data['kpoints'])
        print(f"Number of k-points in data: {len(kpoints)}")

        # save all original data for later plotting for comparison
        if 'kpath' in data:
            self.kpath = np.array(data['kpath'])
        else:
            self.kpath = np.linspace(0, 1, len(kpoints))
        self.bands = np.array(data['bands'])
        self.Sx = np.array(data['Sx'])
        self.Sy = np.array(data['Sy'])
        self.Sz = np.array(data['Sz'])
        if shift_by_fermi is True:
            self.bands -= self.E_F
        n_bands = np.array(data['bands']).shape[1]
        idx_bands = [band+1 in band_numbers for band in range(n_bands)]

        if selection_by_energy is True:
            bands_to_fit = self.bands[:, idx_bands]
              # ndarray of bool values for what bands are within limits
            within_limits = np.logical_and(energy_lim[0] <= bands_to_fit, bands_to_fit <= energy_lim[1])
              # list of bool values for what k-points from dataset to choose
            idx_kpoints = np.any(within_limits, axis=1)
              # ndarray of bool values for what bands are within limits at the k-points which are selected
                # you will need to rule out all 'False' cases when calculating e.g. the model spin texture and then 
                #    fitting
            idx_bands_at_select_kpoints = within_limits[idx_kpoints]
            self.idx_bands_at_select_kpoints = idx_bands_at_select_kpoints
        else:
            # indeces for selection of bands and k-points
            if metric=='Euclidean':
                # Euclidean metric
                within_limits = np.array(np.power(kpoints[:,0], 2) + np.power(kpoints[:,1], 2) <= k_lim**2, dtype=bool)
                idx_kpoints = within_limits      
            elif metric=='Manhattan':
                # Manhattan metric
                within_limits = np.array((kpoints[:,0]>=kx_lims[0], kpoints[:,0]<=kx_lims[1], \
                                        kpoints[:,1]>=ky_lims[0], kpoints[:,1]<=ky_lims[1] ), \
                                            dtype=bool)
                idx_kpoints = np.logical_and.reduce(within_limits)
            else:
                raise ValueError('Only Euclidean and Manhattan metrics are possible for k-points.')
            
            # ndarray of bool values for what bands are within limits at the k-points which are selected
            #   only for compatibility with the 'energy_window' method above
            #   here the (len_kpoints, 2) array is full of True, no False present
            idx_bands_at_select_kpoints = np.full((np.sum(idx_kpoints), 2), fill_value=True)
            self.idx_bands_at_select_kpoints = idx_bands_at_select_kpoints

        # select only required kpoints, bands and spins,
        #   save them to instance variables
          # in case it's 2D calculation, there is (normally) no 'kpath' in data
        self.kpoints = kpoints[idx_kpoints]
        bands_tofit = self.bands[:, idx_bands]
        self.bands_tofit = bands_tofit[idx_kpoints]
        Sx = np.array(data['Sx'])[idx_kpoints, :]
        self.Sx_tofit = Sx[:, idx_bands]
        Sy = np.array(data['Sy'])[idx_kpoints, :]
        self.Sy_tofit = Sy[:, idx_bands]
        Sz = np.array(data['Sz'])[idx_kpoints, :]
        self.Sz_tofit = Sz[:, idx_bands]
        print(f"Number of k-points being fitted: {self.bands_tofit.shape[0]}")

    def model_spin_texture(self, kpoints, params, S_vec=(0,0,1)):
        if self.explicit_E0_mstar_J0 is True:
            E0, m_star, J0, *Rashba_params = params
        else:
            Rashba_params = params
            E0, m_star, J0 = (None, None, None)
        Energies = []
        S_x = []
        S_y = []
        S_z = []
        for k_vec in kpoints:
            H = self.model(k_vec, E0, m_star, J0, S_vec, Rashba_params)
            E, eigvecs = np.linalg.eigh(H)
            Energies.append(np.real(E))
            S_x.append(np.real([np.conj(eigvecs[:,i]).T @ sigma_x_n @ eigvecs[:,i] for i in range(eigvecs.shape[1])]))
            S_y.append(np.real([np.conj(eigvecs[:,i]).T @ sigma_y_n @ eigvecs[:,i] for i in range(eigvecs.shape[1])]))
            S_z.append(np.real([np.conj(eigvecs[:,i]).T @ sigma_z_n @ eigvecs[:,i] for i in range(eigvecs.shape[1])]))
        # convert to numpy arrays
        Energies = np.array(Energies)
        S_x = np.array(S_x)
        S_y = np.array(S_y)
        S_z = np.array(S_z)
        # SORT
        idx_ascend = np.argsort(Energies, axis=1)
        Energies = np.take_along_axis(Energies, idx_ascend, axis=1)
        S_x = np.take_along_axis(S_x, idx_ascend, axis=1)
        S_y = np.take_along_axis(S_y, idx_ascend, axis=1)
        S_z = np.take_along_axis(S_z, idx_ascend, axis=1)
        return Energies, S_x, S_y, S_z

    def define_model_Hamiltonian(self, fun):
        """Take a function with all the terms that you want.

        Args:
            fun (function): accepted arguments are 
                k_vec (real 3x1 array): vector in reciprocal space, units of 1/Angstrom.
                E0 (real): energy at k_vec = (0,0,0), units of eV.
                m_star (real): effective mass relative to m_e, unitless.
                J0 (real): exchange coupling, units of meV.
                S_vec (real 3x1 array): unit vector of direction of magnetization, 
                params (real array): array of parameters of the model, units of eV/Angstrom^n, where n is the order (in k) of the expansion term.
                point_group (string): point group name, defaults to 'C3v'. 
                
        """
        self.model = fun

    def cost_function(self, parameters):
        """cost function to minimize"""
        # 1. get the energies and spin texture
        Energies, S_x, S_y, S_z = self.model_spin_texture(self.kpoints, parameters, S_vec=(0,0,1))
        
        # 2. get the RMSE
        RMSE_E = np.sum(np.power(Energies[self.idx_bands_at_select_kpoints] - self.bands_tofit[self.idx_bands_at_select_kpoints], 2))
        RMSE_Sx = np.sum(np.power(S_x[self.idx_bands_at_select_kpoints] - self.Sx_tofit[self.idx_bands_at_select_kpoints], 2))
        RMSE_Sy = np.sum(np.power(S_y[self.idx_bands_at_select_kpoints] - self.Sy_tofit[self.idx_bands_at_select_kpoints], 2))
        RMSE_Sz = np.sum(np.power(S_z[self.idx_bands_at_select_kpoints] - self.Sz_tofit[self.idx_bands_at_select_kpoints], 2))
        RMSE_spin =  (  weight_RMSE_spin_inplane * (RMSE_Sx + RMSE_Sy) + RMSE_Sz  ) / (weight_RMSE_spin_inplane + 1)
        
        # total RMSE as a sum of energy and spin
        RMSE = RMSE_E + weight_RMSE_spin*RMSE_spin #RMSE_spin #
        return RMSE

    def fit_model_to_bands(self, initial_guess, system='Wannier', tol=1e-10, bounds=None):
        """Fit model to wannier or DFT bands: for all kpoints in self.kpoints calculate the bands and spin texture. 
            Then compare to self.bands_tofit, self.Sx_tofit, self.Sy_tofit, self.Sz_tofit -> calculate the RMSE -> optimize RMSE by adjusting
            parameters.

        Args:
            initial_guess (_type_): _description_
            system (str, optional): Which data to fit to: either 'Wannier' or 'DFT'. Defaults to 'Wannier'.

        """
        self.initial_fit_params = initial_guess

        # 3. optimize parameters
        if self.initial_plot_only is True:
            fit_results = minimize(self.cost_function, x0=initial_guess, bounds=bounds, tol=1e9, options=fit_options) 
        else:
            fit_results = minimize(self.cost_function, x0=initial_guess, bounds=bounds, method = 'Nelder-Mead', options=fit_options) #, tol=tol)
        
        # 4. return the best ones and 
        self.fit_results = fit_results
        self.best_params = fit_results['x']
        self.best_cost   = self.fit_results['fun']/self.bands_tofit.shape[0]

    def get_fit_results(self, system='Wannier'):
        """Get the results of the fit.
        """
        return self.best_params

    def importance_of_parameters(self, refit_at_each=False, exclude_params_below=0.01):
        """Estimate the importance of each parameter by calculating the increase in cost function 
        when each term is removed.
        If refit_at_each == True, then refit the data with the given parameter bound to be zero --- more correct, more expensive.
        Else, just set the parameter to 0 and keep the other parameters as they are. 

        Args:
            refit_at_each (bool, optional): Determines if the model should refit all the other parameters after removing the one. Defaults to False.
            exclude_params_below (float, optional): those terms will be excluded, which, when removed, increase the cost function by less than 'exclude_params_below' 
                                                        (normalized to the highest cost function increase)  

        Returns: 
            the costs when one by one of the parameters is set to zero (and the rest is or is not refitted)
        """
        self.refit_at_each = refit_at_each

        # Cost increase --WITHOUT-- refitting
        costs_no_refit = []
        for i_param in range(len(self.best_params)):
            cropped_params = copy(self.best_params)
            cropped_params[i_param] = 0
            costs_no_refit.append(self.cost_function(cropped_params))
        self.costs_cropped_without_refit = np.array(costs_no_refit)/self.bands_tofit.shape[0]

            # --- REDUCED MODEL --- 
                #   refit with unimportant terms excluded
        # which terms are below the threshold
        exclude_terms = (self.costs_cropped_without_refit / np.max(self.costs_cropped_without_refit)) < exclude_params_below
        bounds_reduced_model = [(-1e4, 1e4)] * len(self.best_params)
        init_params_reduced_model = copy(self.best_params)
        for i, exclude in enumerate(exclude_terms):
            if exclude:
                bounds_reduced_model[i] = (0, 0)
                init_params_reduced_model[i] = 0
        self.initial_fit_params_reduced_model = init_params_reduced_model
        print(f"Number of parameters in the reduced model: {np.sum(~(exclude_terms))} out of {len(self.best_params)} above {exclude_params_below} threshold.")
        fit_results = minimize(self.cost_function, x0=init_params_reduced_model, bounds=bounds_reduced_model, method='Nelder-Mead', options=fit_options) #, tol=tol)   
        self.fit_results_reduced_model = fit_results
        self.best_params_reduced_model = fit_results['x']
        self.best_cost_reduced_model = fit_results['fun']/self.bands_tofit.shape[0]
        self.plot_fit(self.best_params_reduced_model, E_F=self.E_F, E_thr_2D_plot=E_thr_2D_plot, quiver_scale=quiver_scale, fin_1D=fin_1D_name, \
                shift_by_fermi=shift_bands_by_fermi, clim_Sz=color_axis_lim_Sz, clim_Sxy=color_axis_lim_Sxy, name_suffix='_reduced_model')
        self.save_fit_results(fout="best_fit_Rashba_model_reduced.txt", reduced_model=True)
        # sort the terms (their parameters) by importance
        idx_importance_params = np.argsort(self.costs_cropped_without_refit)[::-1]

        # Cost increase --WITH-- refitting
        if refit_at_each is True:
            costs = []
            for i_param in range(len(self.best_params)):
                # bound the i_param-th parameter to be zero
                bounds_i = [(-1e4, 1e4)] * len(self.best_params)
                bounds_i[i_param] = (0, 0)
                # set the i_param-th parameter to zero
                init_params = copy(self.best_params)
                init_params[i_param] = 0
                fit_results = minimize(self.cost_function, x0=init_params, bounds=bounds_i, options=fit_options) #, tol=tol)        
                # 4. return the best ones and 
                self.fit_results = fit_results
                self.best_params = fit_results['x']
                costs.append( fit_results['fun']/self.bands_tofit.shape[0] )

            # to check, recalculate also the 'best cost' .. since we are using bounds, the convergence method may be different and give different cost function
            fit_results = minimize(self.cost_function, x0=init_params, bounds=bounds_i, options=fit_options) #, tol=tol) 
            self.best_cost_with_bounds_check = fit_results['fun']/self.bands_tofit.shape[0]

            # check also with 'Nelder-Mead' to see if it gives exactly the same cost function as before
            fit_results = minimize(self.cost_function, x0=init_params, method='Nelder-Mead', options=fit_options) #, tol=tol) 
            self.best_cost_without_bounds_one_more_check = fit_results['fun']/self.bands_tofit.shape[0]

            # save the costs per k-point
            self.costs_cropped = np.array(costs)/self.bands_tofit.shape[0]

    def plot_fit(self, params, E_F, E_thr_2D_plot=0.02, quiver_scale=1, fin_1D='bands_spin.pickle', shift_by_fermi=True, \
                    clim_Sz=[-1, 1], clim_Sxy=[-1, 1], name_suffix=''):
        """Plot the fitted spin texture at Fermi.

        Args:
            E_F (float): Fermi energy in eV.
            E_thr_2D_plot (float, optional): Energy distance threshold to plot points around selected 
                                                energy value. Units of eV. Defaults to 0.02.
        """
        fig_caption = 'model Hamiltonian'
        fout = f'spin_texture_1D_fit_model{name_suffix}.png'
        Nk = 51
        real_space_lattice_vector_matrix = load_lattice_vectors(win_file="../wannier90.win")
        kpoint_matrix = [
            [(0.00,  0.50,  0.00),    (0.00,  0.00,  0.00)],
            [(0.00,  0.00,  0.00),    (0.67, -0.33,  0.00)],
            [(0.67, -0.33,  0.00),    (0.00,  0.50,  0.00)]
        ]
        G = reciprocal_lattice_vectors(real_space_lattice_vector_matrix)
        kpoints_rec, kpoints_cart, kpath = get_kpoint_path(kpoint_matrix, G, Nk)
        Energies, S_x, S_y, S_z = self.model_spin_texture(kpoints_cart, params, S_vec=(0,0,1))

        # plt.plot(kpath, Energies)
        # plt.ylim([-6, 11])
        # plt.show()
        
        fig, axes = plt.subplots(1, 3, figsize=[11,4.5])
        spin_name = ['Sx', 'Sy', 'Sz']

        if self.calc_2D is True:
            # load 1D data from independent dataset (but coming from the same Wannier model)
            with open(fin_1D, 'rb') as fr:
                data_1D = pickle.load(fr)
            S_vec = [np.array(data_1D['Sx']), np.array(data_1D['Sy']), np.array(data_1D['Sz'])]
            bands = np.array(data_1D['bands'])
            if shift_by_fermi is True:
                bands -= self.E_F
            kpath_plot = np.array(data_1D['kpath'])
        else:
            S_vec = [self.Sx, self.Sy, self.Sz]
            bands = self.bands
            kpath_plot = self.kpath

        # plot original data
        clims = [clim_Sxy, clim_Sxy, clim_Sz]  # color axes
        for i, S in enumerate(S_vec):
            ax = axes[i]
            clim = clims[i]
            ax.axhline(linestyle='--', color='k')
            ax.set_ylim(yaxis_lim)
            ax.set_xlim([min(kpath), max(kpath)])
            ax.set_title(spin_name[i])
            if i == 0:
                ax.set_ylabel('E - E_F (eV)')
            ax.set_xlabel('k-path (1/A)')
            # if ticks is not None and tick_labels is not None:
            #     ax.set_xticks(ticks, tick_labels)
            for j in range(bands.shape[1]):
                sc = ax.scatter(kpath_plot, bands[:,j], marker='o', c=S[:,j], cmap='seismic', s=0.2, vmin=clim[0], vmax=clim[1])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(sc, cax=cax, orientation='vertical')
            #cbar.set_label(r'$S_\mathrm{z}$')
            #sc.set_clim(vmin=colorbar_Sz_lim[0], vmax=colorbar_Sz_lim[1])

        # plot fit 
        for i, S in enumerate([S_x, S_y, S_z]):
            ax = axes[i]
            clim = clims[i]
            ax.set_facecolor("#777777")
            ax.axhline(linestyle='--', color='k')
            ax.set_ylim(yaxis_lim)
            ax.set_xlim([min(kpath), max(kpath)])
            ax.set_title(spin_name[i])
            if i == 0:
                ax.set_ylabel('E - E_F (eV)')
            ax.set_xlabel('k-path (1/A)')
            # if ticks is not None and tick_labels is not None:
            #     ax.set_xticks(ticks, tick_labels)
            for j in range(Energies.shape[1]): 
                sc = ax.scatter(kpath, Energies[:,j], c=S[:,j], marker="$\u25EF$", cmap='seismic', lw=0.5, s=10, vmin=clim[0], vmax=clim[1])
                # sc.set_edgecolors(sc.get_facecolors())
            # sc.set_facecolor('none')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(sc, cax=cax, orientation='vertical')
            #cbar.set_label(r'$S_\mathrm{z}$')
            #sc.set_clim(vmin=colorbar_Sz_lim[0], vmax=colorbar_Sz_lim[1])

        plt.suptitle(fig_caption)
        plt.tight_layout()
        # plt.show()
        
        fout = check_file_exists(fout)
        plt.savefig(fout, dpi=400)
        # plt.close()
        # plt.show()
            
        kpoints_rec2D, kpoints_cart2D = get_2D_kpoint_mesh(G, limits=[-0.5, 0.5], Nk=300)
        kpoints_cart2D = np.array(kpoints_cart2D)
        Energies2D, S_x2D, S_y2D, S_z2D = self.model_spin_texture(kpoints_cart2D, params, S_vec=(0,0,1))

        # plot 2D spin texture
        Es = [0] #1.7] #0.0, 0.4, 0.8, 1.2, 1.6, 2.0] #[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7] #[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        for E in Es:
            fermi_surface_spin_texture(kpoints_cart2D, Energies2D, S_x2D, S_y2D, S_z2D, E=E, E_thr=E_thr_2D_plot, \
                                        E_F=0, fig_name=f"spin_texture_2D{name_suffix}.jpg", \
                                        quiver_scale=quiver_scale, scatter_for_quiver=True, \
                                        scatter_size_quiver=2.0, scatter_size=2.0, reduce_by_factor=1)

    def print_fit_results(self, reduced_model=False, cost_increase_fname="cost_function_increase.dat"):
        """Print best fit parameters.
        """
        if reduced_model is False:
            fit_results = self.fit_results
            params = self.best_params
        elif reduced_model is True:
            fit_results = self.fit_results_reduced_model
            params = self.best_params_reduced_model

        message = ""
        message += f"{fit_results['message']}\ncost function / fitted k-point = {fit_results['fun']/self.bands_tofit.shape[0]}\n"

          # importance of parameters
        if reduced_model is False and hasattr(self, 'costs_cropped_without_refit') :
            message += f"`Cost function/fitted k-point` increase if parameter zeroed (WITHOUT refitting): {repr(self.costs_cropped_without_refit - self.best_cost/self.bands_tofit.shape[0])}\n\n"
            # save cost function increase to a dedicated file
            cost_increase = {}
            cost_increase['no_refitting'] = self.costs_cropped_without_refit - self.best_cost/self.bands_tofit.shape[0]                
            if self.refit_at_each is True:
                message += f"`Cost function/fitted k-point` increase if parameter zeroed (WITH refitting): {repr(self.costs_cropped - self.best_cost/self.bands_tofit.shape[0])}\n\n"
                message += f"Best cost/k-point if all bounds set to (-1e4, 1e4): {repr(self.best_cost_with_bounds_check/self.bands_tofit.shape[0])}\n"
                message += f"Cost/k-point with all parameters and bounds not set (using 'Nelder-Mead'), should give the same as 'best cost': {self.best_cost_without_bounds_one_more_check/self.bands_tofit.shape[0]}\n"
                cost_increase['with_refitting'] = self.costs_cropped - self.best_cost/self.bands_tofit.shape[0]
            # save cost function increase to a dedicated file
            cost_increase_fname = check_file_exists(cost_increase_fname)
            with open(cost_increase_fname, 'a+') as fw:
                fw.write(repr(cost_increase))
        if self.explicit_E0_mstar_J0 is True:
            message += f"""--- BEST FIT PARAMETERS ---\nE0 = {params[0]: .6f} eV
m* = {params[1]: .6f}
J0 = {params[2]: .6f} eV
Rashba params = [{', '.join([str(p) for p in params[3:]])}]"""
        else:
            message += f"Rashba params = [{', '.join([str(p) for p in params])}]"
        return message

    def print_fitted_Hamiltonian(self, system='Wannier'):
        """Print the fitted Hamiltonian as an equation with the best-fit coefficients.

        Args:
            system (str, optional): _description_. Defaults to 'Wannier'.
        """
        raise NotImplementedError

    def save_fit_results(self, fout="best_fit_Rashba_model.txt", reduced_model=False):
        """Save the best fit result to file."""
        if reduced_model is False:
            fit_results = self.fit_results
            initial_fit_params = self.initial_fit_params
        elif reduced_model is True:
            fit_results = self.fit_results_reduced_model
            initial_fit_params = self.initial_fit_params_reduced_model

        best_fit_report = ""
        best_fit_report += f"{fit_results['message']}\ncost function = {fit_results['fun']/self.bands_tofit.shape[0]}\n\n"
        best_fit_report += "HYPERPARAMETERS\n#-----------------------------------\n"
        for name, value in zip(self.hyperparameters_names, self.hyperparameters):
            best_fit_report += f"{name}{' '*(30-len(name))}= {value}\n"
        best_fit_report += f"\n--- INITIAL PARAMETERS ---\n"
        if self.explicit_E0_mstar_J0 is True:
            best_fit_report += f"""E0_init =  {initial_fit_params[0]:.6f} eV
m*_init =  {initial_fit_params[1]:.6f}
J0_init =  {initial_fit_params[2]:.6f} eV\n"""
            best_fit_report += f"Rashba_params_init =  {repr(initial_fit_params[3:])}\n\n"
        else:
            best_fit_report += f"Rashba_params_init =  {repr(initial_fit_params)}\n\n"
        # main resulting fit info, also printed
        best_fit_report += self.print_fit_results(reduced_model=reduced_model)
        # all resulting fit info
        best_fit_report += "\n\n--- ALL FITTING INFO ---\n" + repr(fit_results)
        fout = check_file_exists(fout)
        with open(fout, 'w') as fw:
            fw.write(best_fit_report)

def model_H(k_vec, E0, m_star, J0, S_vec, params, point_group='C3v'):  
        kx, ky, kz = k_vec
        k = np.linalg.norm(np.array(k_vec))
        phi = arctan2(ky, kx)
        
        if point_group == 'plain':
            H = (E0 + C_kinetic/(2*m_star) * k**2) * sigma_0  - J0* (S_vec[0]*sigma_x + S_vec[1]*sigma_y + S_vec[2]*sigma_z)
        
        elif point_group == 'C3v':
            # --> C3v point group
            # 0: linear Rashba
            # 1: cubic Rashba
            # 2: trigonal distortion
            #     # [classical Rashba, only induces hexagon-snowflake opening, induces hexagon from two triangles]
            H = (E0 + C_kinetic/(2*m_star) * k**2) * sigma_0  - J0* (S_vec[0]*sigma_x + S_vec[1]*sigma_y + S_vec[2]*sigma_z) + \
                    (params[0]*k + params[1]*k**3) * (cos(phi)*sigma_y - sin(phi)*sigma_x) + params[2] * k**3 * cos(3*phi) * sigma_z
        
        elif point_group == 'C3v_plus_2nd':
            # --> C3v + second-order point group
            # 0: linear Rashba
            # 1: cubic Rashba
            # 2: trigonal distortion
            # 2nd order term  3: second-order Dresselhaus
            #     # [classical Rashba, only induces hexagon-snowflake opening, induces hexagon from two triangles]
            # 6th order term  4: hexagonal warping
            H = (E0 + C_kinetic/(2*m_star) * k**2) * sigma_0  - J0* (S_vec[0]*sigma_x + S_vec[1]*sigma_y + S_vec[2]*sigma_z) + \
                    (params[0]*k + params[1]*k**3) * (cos(phi)*sigma_y - sin(phi)*sigma_x) + params[2] * k**3 * cos(3*phi) * sigma_z + \
                        params[3] * ( (kx**2-ky**2)*sigma_x - 2*kx*ky*sigma_y ) + \
                            params[4] * k**6 * sin(6*phi) * sigma_z
        
        elif point_group == 'C3':
            # --> C3 point group
                # 0: linear Dresselhaus
                # 1: linear Rashba
                # 2: cubic Dresselhaus
                # 3: cubic Rashba
                # 4: trigonal distortion
                # 5: trigonal distortion 60 deg rotated
            H = (E0 + C_kinetic/(2*m_star) * k**2) * sigma_0  - J0* (S_vec[0]*sigma_x + S_vec[1]*sigma_y + S_vec[2]*sigma_z) + \
                    params[0] * (kx*sigma_x + ky*sigma_y) + \
                    params[1] * (kx*sigma_y - ky*sigma_x) + \
                    params[2] * ((kx**3+kx*ky**2)*sigma_x + (kx**2*ky+ky**3)*sigma_y) + \
                    params[3] * ((kx**3+kx*ky**2)*sigma_y - (kx**2*ky+ky**3)*sigma_x) + \
                    params[4] * (kx**3-3*kx*ky**2)*sigma_z + \
                    params[5] * (ky**3-3*kx**2*ky)*sigma_z

        elif point_group == 'C3h':
            # --> C3v + second-order point group
            # 0: trigonal distortion
            # 2nd order term  1: second-order Dresselhaus
            #     # [classical Rashba, only induces hexagon-snowflake opening, induces hexagon from two triangles]
            # 6th order term  2: hexagonal warping
            H = (E0 + C_kinetic/(2*m_star) * k**2) * sigma_0  - J0* (S_vec[0]*sigma_x + S_vec[1]*sigma_y + S_vec[2]*sigma_z) + \
                    params[0] * k**3 * cos(3*phi) * sigma_z + \
                        params[1] * ( (kx**2-ky**2)*sigma_x - 2*kx*ky*sigma_y ) + \
                            params[2] * k**6 * sin(6*phi) * sigma_z

        elif point_group == 'C3h_empirical_4th':
            # --> C3v + second-order point group
            # 0: trigonal distortion
            # 2nd order term  1: fourth-order Dresselhaus
            #     # [classical Rashba, only induces hexagon-snowflake opening, induces hexagon from two triangles]
            # 6th order term  2: hexagonal warping
            H = (E0 + C_kinetic/(2*m_star) * k**2) * sigma_0  - J0* (S_vec[0]*sigma_x + S_vec[1]*sigma_y + S_vec[2]*sigma_z) + \
                        params[0] * k**4 * ( sin(4*phi)*sigma_x - cos(4*phi)*sigma_y ) + \
                            params[1] * k**6 * cos(6*phi) * sigma_z

        elif point_group == 'C3h_empirical_2nd':
            # --> C3v + second-order point group
            # 0: trigonal distortion
            # 2nd order term  1: second-order Dresselhaus
            #     # [classical Rashba, only induces hexagon-snowflake opening, induces hexagon from two triangles]
            # 6th order term  2: hexagonal warping
            H = (E0 + C_kinetic/(2*m_star) * k**2) * sigma_0  - J0* (S_vec[0]*sigma_x + S_vec[1]*sigma_y + S_vec[2]*sigma_z) + \
                        params[0] * k**2 * ( sin(2*phi)*sigma_x + cos(2*phi)*sigma_y ) + \
                            params[1] * k**6 * cos(6*phi) * sigma_z

        elif point_group == 'C3h_empirical_2nd_and_4th':
            # --> C3v + second-order point group
            # 0: trigonal distortion
            # 2nd order term  1: second-order Dresselhaus
            # 4th order term  2: fourth-order Dresselhaus
            # 6th order term  3: hexagonal warping
            H = (E0 + C_kinetic/(2*m_star) * k**2) * sigma_0  - J0* (S_vec[0]*sigma_x + S_vec[1]*sigma_y + S_vec[2]*sigma_z) + \
                        params[0] * k**2 * ( sin(2*phi)*sigma_x + cos(2*phi)*sigma_y ) + \
                            params[1] * k**4 *  ( sin(4*phi)*sigma_x - cos(4*phi)*sigma_y ) + \
                            params[2] * k**6 * cos(6*phi) * sigma_z

        elif point_group == '-3m_prime':
            # --> 3m_prime magnetic point group derived from D_3d (-3m)
            # terms in the order as they are in expansion EXCEPT constant term and the k^2 term, which would be obsolete (already kinetic energy there)
            H = (E0 + C_kinetic/(2*m_star) * k**2) * sigma_0  - J0* (S_vec[0]*sigma_x + S_vec[1]*sigma_y + S_vec[2]*sigma_z) + \
                            (params[0] * k**4 + params[1] * k**6 + params[2] * k**6 * cos(6*phi)) * sigma_0 + \
                            (params[3] * k**2 + params[4] * k**4 + params[5] * k**6 + params[6] * k**6 * cos(6*phi) ) * sigma_z + \
                            (params[7]*k**2 + params[8] * k**4) * (-sin(2*phi)*sigma_x - cos(2*phi)*sigma_y ) + \
                            params[9] * k**4 *  ( -sin(4*phi)*sigma_x + cos(4*phi)*sigma_y ) + \
                            params[10] * k**6 * ( (sin(2*phi)-sin(4*phi))*sigma_x + (cos(2*phi)+cos(4*phi))*sigma_y ) + \
                            params[11] * k**6 * ( (sin(2*phi)+sin(4*phi))*sigma_x + (cos(2*phi)-cos(4*phi))*sigma_y )


        elif point_group == '-3m_prime_p_type':
            # --> 3m_prime magnetic point group derived from D_3d (-3m)
            # terms in the order as they are in expansion EXCEPT constant term and the k^2 term, which would be obsolete (already kinetic energy there)
            H = outer(      (E0 + C_kinetic/(2*m_star) * k**2) * sigma_0  - J0* (S_vec[0]*sigma_x + S_vec[1]*sigma_y + S_vec[2]*sigma_z) + \
                            (params[0] * k**4 + params[1] * k**6 + params[2] * k**6 * cos(6*phi) ) * sigma_0 + \
                            (params[3] + params[4] * k**2 + params[5] * k**4 + params[6] * k**6 + params[7] * k**6 * cos(6*phi) ) * sigma_z, rho_0) + \
                outer(      (params[8] + params[9] * k**2 + params[10] * k**4 ) * sigma_0, rho_y) + \
                outer(      (params[11] + params[12] * k**2 + params[13] * k**4 ) * sigma_z, rho_y) + \
                            (params[14] + params[15] * k**2 + params[16] * k**4 + params[17] * k**6 + params[18] * k**6 * cos(6*phi) ) * (outer(sigma_x, rho_x) + outer(sigma_y, rho_z)) + \
                k**2 * params[19] * (outer(sin(2*phi)*sigma_0, rho_x) + outer(cos(2*phi)*sigma_0, rho_z)) + \
                k**2 * params[20] * (outer(sin(2*phi)*sigma_z, rho_x) + outer(cos(2*phi)*sigma_z, rho_z)) + \
                k**2 * params[21] * (outer(sin(2*phi)*sigma_z, rho_0) + outer(cos(2*phi)*sigma_y, rho_0)) + \
                k**2 * params[22] * (outer(sin(2*phi)*sigma_x, rho_y) + outer(cos(2*phi)*sigma_y, rho_y)) + \
                k**2 * params[23] * (sin(2*phi)*(outer(sigma_y, rho_x) + outer(sigma_x, rho_z)) + cos(2*phi)*(outer(sigma_x, rho_x) - outer(sigma_y, rho_z))) +\
                k**4 * params[24] * ()
                            

        else:
            raise ValueError('The only point groups implemented are C3 and C3v so far.')
        return H


def model_min3mprime_p(k_vec, E0, m_star, J0, S_vec, params):
    """D3d + Mz"""
    k_x, k_y, k_z = k_vec
    ## model with 5 parameters for orders [0] + kinetic energy
    # return params[0]*sig_rho['s_0*rho_0'] + params[1]*sig_rho['s_0*rho_y'] + params[2]*sig_rho['s_x*rho_x'] + params[2]*sig_rho['s_y*rho_z'] + params[3]*sig_rho['s_z*rho_0'] + params[4]*sig_rho['s_z*rho_y'] + sig_rho['s_0*rho_0']*(k_x**2*params[5] + k_y**2*params[5])
    ## model with 15 parameters for orders [0, 2]
    # return 2*k_x*k_y*params[10]*sig_rho['s_x*rho_y'] + 2*k_x*k_y*params[14]*sig_rho['s_z*rho_x'] + 2*k_x*k_y*params[7]*sig_rho['s_0*rho_x'] + 2*k_x*k_y*params[9]*sig_rho['s_x*rho_0'] + sig_rho['s_0*rho_0']*(k_x**2*params[5] + k_y**2*params[5] + params[0]) + sig_rho['s_0*rho_y']*(k_x**2*params[6] + k_y**2*params[6] + params[1]) + sig_rho['s_0*rho_z']*(k_x**2*params[7] - k_y**2*params[7]) + sig_rho['s_x*rho_x']*(k_x**2*params[8] + k_y**2*params[11] + params[2]) + sig_rho['s_x*rho_z']*(-k_x*k_y*params[11] + k_x*k_y*params[8]) + sig_rho['s_y*rho_0']*(k_x**2*params[9] - k_y**2*params[9]) + sig_rho['s_y*rho_x']*(-k_x*k_y*params[11] + k_x*k_y*params[8]) + sig_rho['s_y*rho_y']*(k_x**2*params[10] - k_y**2*params[10]) + sig_rho['s_y*rho_z']*(k_x**2*params[11] + k_y**2*params[8] + params[2]) + sig_rho['s_z*rho_0']*(k_x**2*params[12] + k_y**2*params[12] + params[3]) + sig_rho['s_z*rho_y']*(k_x**2*params[13] + k_y**2*params[13] + params[4]) + sig_rho['s_z*rho_z']*(k_x**2*params[14] - k_y**2*params[14])
    ## model with 51 parameters for orders [0, 1, 2, 3, 4, 5, 6]
    return sig_rho['s_0*rho_0']*(3*k_x**6*params[30] - 18*k_x**4*k_y**2*params[30] + 27*k_x**4*k_y**2*params[46] + 3*k_x**4*params[15] + 27*k_x**2*k_y**4*params[30] - 18*k_x**2*k_y**4*params[46] + 6*k_x**2*k_y**2*params[15] + 3*k_x**2*params[5] + 3*k_y**6*params[46] + 3*k_y**4*params[15] + 3*k_y**2*params[5] + 3*params[0])/3 + sig_rho['s_0*rho_x']*(3*k_x**5*k_y*params[32] - 9*k_x**5*k_y*params[40] - 6*k_x**3*k_y**3*params[32] - 6*k_x**3*k_y**3*params[40] + 3*k_x**3*k_y*params[17] - 9*k_x**3*k_y*params[25] - 9*k_x*k_y**5*params[32] + 3*k_x*k_y**5*params[40] - 9*k_x*k_y**3*params[17] + 3*k_x*k_y**3*params[25] + 6*k_x*k_y*params[7])/3 + sig_rho['s_0*rho_y']*(3*k_x**6*params[31] - 18*k_x**4*k_y**2*params[31] + 27*k_x**4*k_y**2*params[47] + 3*k_x**4*params[16] + 27*k_x**2*k_y**4*params[31] - 18*k_x**2*k_y**4*params[47] + 6*k_x**2*k_y**2*params[16] + 3*k_x**2*params[6] + 3*k_y**6*params[47] + 3*k_y**4*params[16] + 3*k_y**2*params[6] + 3*params[1])/3 + sig_rho['s_0*rho_z']*(-3*k_x**6*params[32] + 6*k_x**4*k_y**2*params[32] - 9*k_x**4*k_y**2*params[40] - 3*k_x**4*params[17] + 9*k_x**2*k_y**4*params[32] - 6*k_x**2*k_y**4*params[40] + 9*k_x**2*k_y**2*params[17] - 9*k_x**2*k_y**2*params[25] + 3*k_x**2*params[7] + 3*k_y**6*params[40] + 3*k_y**4*params[25] - 3*k_y**2*params[7])/3 + sig_rho['s_x*rho_0']*(3*k_x**5*k_y*params[34] - 9*k_x**5*k_y*params[41] - 6*k_x**3*k_y**3*params[34] - 6*k_x**3*k_y**3*params[41] - 3*k_x**3*k_y*params[19] - 9*k_x**3*k_y*params[26] - 9*k_x*k_y**5*params[34] + 3*k_x*k_y**5*params[41] + 9*k_x*k_y**3*params[19] + 3*k_x*k_y**3*params[26] + 6*k_x*k_y*params[9])/3 + sig_rho['s_x*rho_x']*(3*k_x**6*params[33] - 18*k_x**4*k_y**2*params[33] + 6*k_x**4*k_y**2*params[36] + 9*k_x**4*k_y**2*params[44] + 27*k_x**4*k_y**2*params[48] + 3*k_x**4*params[18] + 27*k_x**2*k_y**4*params[33] - 18*k_x**2*k_y**4*params[36] - 12*k_x**2*k_y**4*params[44] - 18*k_x**2*k_y**4*params[48] - 9*k_x**2*k_y**2*params[18] + 3*k_x**2*k_y**2*params[21] - 9*k_x**2*k_y**2*params[28] + 3*k_x**2*params[8] + 3*k_y**6*params[44] + 3*k_y**6*params[48] + 3*k_y**4*params[21] + 3*k_y**4*params[28] + 3*k_y**2*params[11] + 3*params[2])/3 + sig_rho['s_x*rho_y']*(3*k_x**5*k_y*params[35] - 9*k_x**5*k_y*params[42] - 6*k_x**3*k_y**3*params[35] - 6*k_x**3*k_y**3*params[42] - 3*k_x**3*k_y*params[20] - 9*k_x**3*k_y*params[27] - 9*k_x*k_y**5*params[35] + 3*k_x*k_y**5*params[42] + 9*k_x*k_y**3*params[20] + 3*k_x*k_y**3*params[27] + 6*k_x*k_y*params[10])/3 + sig_rho['s_x*rho_z']*(3*k_x**5*k_y*params[36] + 3*k_x**5*k_y*params[43] - 12*k_x**3*k_y**3*params[36] - 10*k_x**3*k_y**3*params[43] - 18*k_x**3*k_y**3*params[44] - 3*k_x**3*k_y*params[18] - 3*k_x**3*k_y*params[21] - 9*k_x**3*k_y*params[28] + 9*k_x*k_y**5*params[36] + 3*k_x*k_y**5*params[43] + 6*k_x*k_y**5*params[44] + 9*k_x*k_y**3*params[18] - 3*k_x*k_y**3*params[21] + 3*k_x*k_y**3*params[28] - 3*k_x*k_y*params[11] + 3*k_x*k_y*params[8])/3 + sig_rho['s_y*rho_0']*(-3*k_x**6*params[34] + 6*k_x**4*k_y**2*params[34] - 9*k_x**4*k_y**2*params[41] + 3*k_x**4*params[19] + 9*k_x**2*k_y**4*params[34] - 6*k_x**2*k_y**4*params[41] - 9*k_x**2*k_y**2*params[19] - 9*k_x**2*k_y**2*params[26] + 3*k_x**2*params[9] + 3*k_y**6*params[41] + 3*k_y**4*params[26] - 3*k_y**2*params[9])/3 + sig_rho['s_y*rho_x']*(-6*k_x**5*k_y*params[36] - 3*k_x**5*k_y*params[43] - 9*k_x**5*k_y*params[44] + 18*k_x**3*k_y**3*params[36] + 10*k_x**3*k_y**3*params[43] + 12*k_x**3*k_y**3*params[44] - 3*k_x**3*k_y*params[18] - 3*k_x**3*k_y*params[21] - 9*k_x**3*k_y*params[28] - 3*k_x*k_y**5*params[43] - 3*k_x*k_y**5*params[44] + 9*k_x*k_y**3*params[18] - 3*k_x*k_y**3*params[21] + 3*k_x*k_y**3*params[28] - 3*k_x*k_y*params[11] + 3*k_x*k_y*params[8])/3 + sig_rho['s_y*rho_y']*(-3*k_x**6*params[35] + 6*k_x**4*k_y**2*params[35] - 9*k_x**4*k_y**2*params[42] + 3*k_x**4*params[20] + 9*k_x**2*k_y**4*params[35] - 6*k_x**2*k_y**4*params[42] - 9*k_x**2*k_y**2*params[20] - 9*k_x**2*k_y**2*params[27] + 3*k_x**2*params[10] + 3*k_y**6*params[42] + 3*k_y**4*params[27] - 3*k_y**2*params[10])/3 + sig_rho['s_y*rho_z']*(3*k_x**6*params[33] - 3*k_x**6*params[36] - 18*k_x**4*k_y**2*params[33] + 12*k_x**4*k_y**2*params[36] + 18*k_x**4*k_y**2*params[44] + 27*k_x**4*k_y**2*params[48] - 3*k_x**4*params[18] + 3*k_x**4*params[21] + 27*k_x**2*k_y**4*params[33] - 9*k_x**2*k_y**4*params[36] - 6*k_x**2*k_y**4*params[44] - 18*k_x**2*k_y**4*params[48] + 9*k_x**2*k_y**2*params[18] + 3*k_x**2*k_y**2*params[21] + 9*k_x**2*k_y**2*params[28] + 3*k_x**2*params[11] + 3*k_y**6*params[48] - 3*k_y**4*params[28] + 3*k_y**2*params[8] + 3*params[2])/3 + sig_rho['s_z*rho_0']*(3*k_x**6*params[37] - 18*k_x**4*k_y**2*params[37] + 27*k_x**4*k_y**2*params[49] + 3*k_x**4*params[22] + 27*k_x**2*k_y**4*params[37] - 18*k_x**2*k_y**4*params[49] + 6*k_x**2*k_y**2*params[22] + 3*k_x**2*params[12] + 3*k_y**6*params[49] + 3*k_y**4*params[22] + 3*k_y**2*params[12] + 3*params[3])/3 + sig_rho['s_z*rho_x']*(3*k_x**5*k_y*params[39] - 9*k_x**5*k_y*params[45] - 6*k_x**3*k_y**3*params[39] - 6*k_x**3*k_y**3*params[45] - 3*k_x**3*k_y*params[24] - 9*k_x**3*k_y*params[29] - 9*k_x*k_y**5*params[39] + 3*k_x*k_y**5*params[45] + 9*k_x*k_y**3*params[24] + 3*k_x*k_y**3*params[29] + 6*k_x*k_y*params[14])/3 + sig_rho['s_z*rho_y']*(3*k_x**6*params[38] - 18*k_x**4*k_y**2*params[38] + 27*k_x**4*k_y**2*params[50] + 3*k_x**4*params[23] + 27*k_x**2*k_y**4*params[38] - 18*k_x**2*k_y**4*params[50] + 6*k_x**2*k_y**2*params[23] + 3*k_x**2*params[13] + 3*k_y**6*params[50] + 3*k_y**4*params[23] + 3*k_y**2*params[13] + 3*params[4])/3 + sig_rho['s_z*rho_z']*(-3*k_x**6*params[39] + 6*k_x**4*k_y**2*params[39] - 9*k_x**4*k_y**2*params[45] + 3*k_x**4*params[24] + 9*k_x**2*k_y**4*params[39] - 6*k_x**2*k_y**4*params[45] - 9*k_x**2*k_y**2*params[24] - 9*k_x**2*k_y**2*params[29] + 3*k_x**2*params[14] + 3*k_y**6*params[45] + 3*k_y**4*params[29] - 3*k_y**2*params[14])/3


def model_3mprime_p(k_vec, E0, m_star, J0, S_vec, params):
    """C3v + Mz"""
    k_x, k_y, k_z = k_vec
    # model with 83 parameters for orders [0, 1, 2, 3, 4, 5, 6]
    return sig_rho['s_0*rho_0']*(3*k_x**6*params[62] + 3*k_x**5*params[46] - 18*k_x**4*k_y**2*params[62] + 27*k_x**4*k_y**2*params[78] + 3*k_x**4*params[31] - 6*k_x**3*k_y**2*params[46] + 3*k_x**3*params[20] + 27*k_x**2*k_y**4*params[62] - 18*k_x**2*k_y**4*params[78] + 6*k_x**2*k_y**2*params[31] + 3*k_x**2*params[10] - 9*k_x*k_y**4*params[46] - 9*k_x*k_y**2*params[20] + 3*k_y**6*params[78] + 3*k_y**4*params[31] + 3*k_y**2*params[10] + 3*params[0])/3 + sig_rho['s_0*rho_x']*(-3*k_x**5*k_y*params[64] - 9*k_x**5*k_y*params[72] + 6*k_x**4*k_y*params[48] + 9*k_x**4*k_y*params[56] + 6*k_x**3*k_y**3*params[64] - 6*k_x**3*k_y**3*params[72] - 3*k_x**3*k_y*params[33] - 9*k_x**3*k_y*params[41] - 18*k_x**2*k_y**3*params[48] - 12*k_x**2*k_y**3*params[56] - 3*k_x**2*k_y*params[22] + 9*k_x*k_y**5*params[64] + 3*k_x*k_y**5*params[72] + 9*k_x*k_y**3*params[33] + 3*k_x*k_y**3*params[41] + 6*k_x*k_y*params[12] + 3*k_y**5*params[56] - 3*k_y**3*params[22] - 3*k_y*params[5])/3 + sig_rho['s_0*rho_y']*(3*k_x**6*params[63] + 3*k_x**5*params[47] - 18*k_x**4*k_y**2*params[63] + 27*k_x**4*k_y**2*params[79] + 3*k_x**4*params[32] - 6*k_x**3*k_y**2*params[47] + 3*k_x**3*params[21] + 27*k_x**2*k_y**4*params[63] - 18*k_x**2*k_y**4*params[79] + 6*k_x**2*k_y**2*params[32] + 3*k_x**2*params[11] - 9*k_x*k_y**4*params[47] - 9*k_x*k_y**2*params[21] + 3*k_y**6*params[79] + 3*k_y**4*params[32] + 3*k_y**2*params[11] + 3*params[1])/3 + sig_rho['s_0*rho_z']*(3*k_x**6*params[64] + 3*k_x**5*params[48] - 6*k_x**4*k_y**2*params[64] - 9*k_x**4*k_y**2*params[72] + 3*k_x**4*params[33] - 12*k_x**3*k_y**2*params[48] - 18*k_x**3*k_y**2*params[56] + 3*k_x**3*params[22] - 9*k_x**2*k_y**4*params[64] - 6*k_x**2*k_y**4*params[72] - 9*k_x**2*k_y**2*params[33] - 9*k_x**2*k_y**2*params[41] + 3*k_x**2*params[12] + 9*k_x*k_y**4*params[48] + 6*k_x*k_y**4*params[56] + 3*k_x*k_y**2*params[22] + 3*k_x*params[5] + 3*k_y**6*params[72] + 3*k_y**4*params[41] - 3*k_y**2*params[12])/3 + sig_rho['s_x*rho_0']*(-3*k_x**5*k_y*params[66] - 9*k_x**5*k_y*params[73] + 6*k_x**4*k_y*params[50] + 9*k_x**4*k_y*params[57] + 6*k_x**3*k_y**3*params[66] - 6*k_x**3*k_y**3*params[73] - 3*k_x**3*k_y*params[35] - 9*k_x**3*k_y*params[42] - 18*k_x**2*k_y**3*params[50] - 12*k_x**2*k_y**3*params[57] - 3*k_x**2*k_y*params[24] + 9*k_x*k_y**5*params[66] + 3*k_x*k_y**5*params[73] + 9*k_x*k_y**3*params[35] + 3*k_x*k_y**3*params[42] + 6*k_x*k_y*params[14] + 3*k_y**5*params[57] - 3*k_y**3*params[24] - 3*k_y*params[7])/3 + sig_rho['s_x*rho_x']*(3*k_x**6*params[65] + 3*k_x**5*params[49] - 18*k_x**4*k_y**2*params[65] - 6*k_x**4*k_y**2*params[68] - 9*k_x**4*k_y**2*params[76] + 27*k_x**4*k_y**2*params[80] + 3*k_x**4*params[34] - 9*k_x**3*k_y**2*params[49] + 3*k_x**3*k_y**2*params[52] + 9*k_x**3*k_y**2*params[59] + 9*k_x**3*k_y**2*params[60] + 3*k_x**3*params[23] + 27*k_x**2*k_y**4*params[65] + 18*k_x**2*k_y**4*params[68] + 12*k_x**2*k_y**4*params[76] - 18*k_x**2*k_y**4*params[80] - 9*k_x**2*k_y**2*params[34] + 3*k_x**2*k_y**2*params[37] - 9*k_x**2*k_y**2*params[44] + 3*k_x**2*params[13] - 9*k_x*k_y**4*params[52] - 3*k_x*k_y**4*params[59] - 3*k_x*k_y**4*params[60] - 9*k_x*k_y**2*params[23] + 6*k_x*k_y**2*params[26] + 3*k_x*params[6] - 3*k_y**6*params[76] + 3*k_y**6*params[80] + 3*k_y**4*params[37] + 3*k_y**4*params[44] + 3*k_y**2*params[16] + 3*params[2])/3 + sig_rho['s_x*rho_y']*(-3*k_x**5*k_y*params[67] - 9*k_x**5*k_y*params[74] + 6*k_x**4*k_y*params[51] + 9*k_x**4*k_y*params[58] + 6*k_x**3*k_y**3*params[67] - 6*k_x**3*k_y**3*params[74] + 3*k_x**3*k_y*params[36] - 9*k_x**3*k_y*params[43] - 18*k_x**2*k_y**3*params[51] - 12*k_x**2*k_y**3*params[58] - 3*k_x**2*k_y*params[25] + 9*k_x*k_y**5*params[67] + 3*k_x*k_y**5*params[74] - 9*k_x*k_y**3*params[36] + 3*k_x*k_y**3*params[43] + 6*k_x*k_y*params[15] + 3*k_y**5*params[58] - 3*k_y**3*params[25] - 3*k_y*params[8])/3 + sig_rho['s_x*rho_z']*(-3*k_x**5*k_y*params[68] + 3*k_x**5*k_y*params[75] + 3*k_x**4*k_y*params[49] - 3*k_x**4*k_y*params[52] - 9*k_x**4*k_y*params[59] + 12*k_x**3*k_y**3*params[68] - 10*k_x**3*k_y**3*params[75] + 18*k_x**3*k_y**3*params[76] - 3*k_x**3*k_y*params[34] - 3*k_x**3*k_y*params[37] - 9*k_x**3*k_y*params[44] - 9*k_x**2*k_y**3*params[49] + 9*k_x**2*k_y**3*params[52] + 3*k_x**2*k_y**3*params[59] + 9*k_x**2*k_y**3*params[60] + 3*k_x**2*k_y*params[26] - 9*k_x**2*k_y*params[30] - 9*k_x*k_y**5*params[68] + 3*k_x*k_y**5*params[75] - 6*k_x*k_y**5*params[76] + 9*k_x*k_y**3*params[34] - 3*k_x*k_y**3*params[37] + 3*k_x*k_y**3*params[44] + 3*k_x*k_y*params[13] - 3*k_x*k_y*params[16] - 3*k_y**5*params[60] - 3*k_y**3*params[26] + 3*k_y**3*params[30] - 3*k_y*params[6])/3 + sig_rho['s_y*rho_0']*(3*k_x**6*params[66] + 3*k_x**5*params[50] - 6*k_x**4*k_y**2*params[66] - 9*k_x**4*k_y**2*params[73] + 3*k_x**4*params[35] - 12*k_x**3*k_y**2*params[50] - 18*k_x**3*k_y**2*params[57] + 3*k_x**3*params[24] - 9*k_x**2*k_y**4*params[66] - 6*k_x**2*k_y**4*params[73] - 9*k_x**2*k_y**2*params[35] - 9*k_x**2*k_y**2*params[42] + 3*k_x**2*params[14] + 9*k_x*k_y**4*params[50] + 6*k_x*k_y**4*params[57] + 3*k_x*k_y**2*params[24] + 3*k_x*params[7] + 3*k_y**6*params[73] + 3*k_y**4*params[42] - 3*k_y**2*params[14])/3 + sig_rho['s_y*rho_x']*(6*k_x**5*k_y*params[68] - 3*k_x**5*k_y*params[75] + 9*k_x**5*k_y*params[76] + 3*k_x**4*k_y*params[49] - 3*k_x**4*k_y*params[52] - 9*k_x**4*k_y*params[60] - 18*k_x**3*k_y**3*params[68] + 10*k_x**3*k_y**3*params[75] - 12*k_x**3*k_y**3*params[76] - 3*k_x**3*k_y*params[34] - 3*k_x**3*k_y*params[37] - 9*k_x**3*k_y*params[44] - 9*k_x**2*k_y**3*params[49] + 9*k_x**2*k_y**3*params[52] + 9*k_x**2*k_y**3*params[59] + 3*k_x**2*k_y**3*params[60] - 6*k_x**2*k_y*params[26] + 9*k_x**2*k_y*params[30] - 3*k_x*k_y**5*params[75] + 3*k_x*k_y**5*params[76] + 9*k_x*k_y**3*params[34] - 3*k_x*k_y**3*params[37] + 3*k_x*k_y**3*params[44] + 3*k_x*k_y*params[13] - 3*k_x*k_y*params[16] - 3*k_y**5*params[59] - 3*k_y**3*params[30] - 3*k_y*params[6])/3 + sig_rho['s_y*rho_y']*(3*k_x**6*params[67] + 3*k_x**5*params[51] - 6*k_x**4*k_y**2*params[67] - 9*k_x**4*k_y**2*params[74] - 3*k_x**4*params[36] - 12*k_x**3*k_y**2*params[51] - 18*k_x**3*k_y**2*params[58] + 3*k_x**3*params[25] - 9*k_x**2*k_y**4*params[67] - 6*k_x**2*k_y**4*params[74] + 9*k_x**2*k_y**2*params[36] - 9*k_x**2*k_y**2*params[43] + 3*k_x**2*params[15] + 9*k_x*k_y**4*params[51] + 6*k_x*k_y**4*params[58] + 3*k_x*k_y**2*params[25] + 3*k_x*params[8] + 3*k_y**6*params[74] + 3*k_y**4*params[43] - 3*k_y**2*params[15])/3 + sig_rho['s_y*rho_z']*(3*k_x**6*params[65] + 3*k_x**6*params[68] + 3*k_x**5*params[52] - 18*k_x**4*k_y**2*params[65] - 12*k_x**4*k_y**2*params[68] - 18*k_x**4*k_y**2*params[76] + 27*k_x**4*k_y**2*params[80] - 3*k_x**4*params[34] + 3*k_x**4*params[37] + 3*k_x**3*k_y**2*params[49] - 9*k_x**3*k_y**2*params[52] - 9*k_x**3*k_y**2*params[59] - 9*k_x**3*k_y**2*params[60] + 3*k_x**3*params[23] - 3*k_x**3*params[26] + 27*k_x**2*k_y**4*params[65] + 9*k_x**2*k_y**4*params[68] + 6*k_x**2*k_y**4*params[76] - 18*k_x**2*k_y**4*params[80] + 9*k_x**2*k_y**2*params[34] + 3*k_x**2*k_y**2*params[37] + 9*k_x**2*k_y**2*params[44] + 3*k_x**2*params[16] - 9*k_x*k_y**4*params[49] + 3*k_x*k_y**4*params[59] + 3*k_x*k_y**4*params[60] - 9*k_x*k_y**2*params[23] + 3*k_x*k_y**2*params[26] - 3*k_x*params[6] + 3*k_y**6*params[80] - 3*k_y**4*params[44] + 3*k_y**2*params[13] + 3*params[2])/3 + sig_rho['s_z*rho_0']*(3*k_x**6*params[69] + 3*k_x**5*params[53] - 18*k_x**4*k_y**2*params[69] + 27*k_x**4*k_y**2*params[81] + 3*k_x**4*params[38] - 6*k_x**3*k_y**2*params[53] + 3*k_x**3*params[27] + 27*k_x**2*k_y**4*params[69] - 18*k_x**2*k_y**4*params[81] + 6*k_x**2*k_y**2*params[38] + 3*k_x**2*params[17] - 9*k_x*k_y**4*params[53] - 9*k_x*k_y**2*params[27] + 3*k_y**6*params[81] + 3*k_y**4*params[38] + 3*k_y**2*params[17] + 3*params[3])/3 + sig_rho['s_z*rho_x']*(-3*k_x**5*k_y*params[71] - 9*k_x**5*k_y*params[77] + 6*k_x**4*k_y*params[55] + 9*k_x**4*k_y*params[61] + 6*k_x**3*k_y**3*params[71] - 6*k_x**3*k_y**3*params[77] - 3*k_x**3*k_y*params[40] - 9*k_x**3*k_y*params[45] - 18*k_x**2*k_y**3*params[55] - 12*k_x**2*k_y**3*params[61] - 3*k_x**2*k_y*params[29] + 9*k_x*k_y**5*params[71] + 3*k_x*k_y**5*params[77] + 9*k_x*k_y**3*params[40] + 3*k_x*k_y**3*params[45] + 6*k_x*k_y*params[19] + 3*k_y**5*params[61] - 3*k_y**3*params[29] - 3*k_y*params[9])/3 + sig_rho['s_z*rho_y']*(3*k_x**6*params[70] + 3*k_x**5*params[54] - 18*k_x**4*k_y**2*params[70] + 27*k_x**4*k_y**2*params[82] + 3*k_x**4*params[39] - 6*k_x**3*k_y**2*params[54] + 3*k_x**3*params[28] + 27*k_x**2*k_y**4*params[70] - 18*k_x**2*k_y**4*params[82] + 6*k_x**2*k_y**2*params[39] + 3*k_x**2*params[18] - 9*k_x*k_y**4*params[54] - 9*k_x*k_y**2*params[28] + 3*k_y**6*params[82] + 3*k_y**4*params[39] + 3*k_y**2*params[18] + 3*params[4])/3 + sig_rho['s_z*rho_z']*(3*k_x**6*params[71] + 3*k_x**5*params[55] - 6*k_x**4*k_y**2*params[71] - 9*k_x**4*k_y**2*params[77] + 3*k_x**4*params[40] - 12*k_x**3*k_y**2*params[55] - 18*k_x**3*k_y**2*params[61] + 3*k_x**3*params[29] - 9*k_x**2*k_y**4*params[71] - 6*k_x**2*k_y**4*params[77] - 9*k_x**2*k_y**2*params[40] - 9*k_x**2*k_y**2*params[45] + 3*k_x**2*params[19] + 9*k_x*k_y**4*params[55] + 6*k_x*k_y**4*params[61] + 3*k_x*k_y**2*params[29] + 3*k_x*params[9] + 3*k_y**6*params[77] + 3*k_y**4*params[45] - 3*k_y**2*params[19])/3


def convert_params(old_params, positions_in_new_list, num_params_new):
    """Convert converged parameters from a high-symmetry model to a lower-symmetry model, which contains these parameters
     but in addition also some others.

    Args:
        old_params (list of floats): list of parameters from the old model
        positions_in_new_list (list of integers): for instance [5, 3, 9] would mean that the 0 parameter from smaller model correpsonds 
                to the 5th parameter of the larger model, the 1st corresponds to the new 3rd, etc. 
        num_params_new (int, optional): Number of parameters in the new (low-symmetry, larger) parameter list. Defaults to 70.
    """
    new_params = [0]*num_params_new
    for i_old, i_new in enumerate(positions_in_new_list):
        new_params[i_new] = old_params[i_old]
    return new_params


def fit():
    global weight_RMSE_spin, yaxis_lim, weight_RMSE_spin_inplane, fit_options, E_thr_2D_plot, quiver_scale, fin_1D_name, shift_bands_by_fermi, color_axis_lim_Sz, color_axis_lim_Sxy

    # =============================== USER INPUT =====================================
    # ================================================================================



    fin_name          = 'bands_spin_2D_30x30.pickle' #'bands_spin_2D_30x30.pickle' #'bands_spin_2D_30x30.pickle' #'bands_spin_30.pickle' #'bands_spin_2D_30x30.pickle' #'bands_spin.pickle' #'bands_spin_CrSTe_U2.10_M001_E0.00.pickle' #'bands_spin_2D_30x30_CrSTe_U2.10_M001_E0.00.pickle' #'bands_spin_CrSTe_U2.10_M001_E0.00.pickle' #'bands_spin_2D_30x30_EFIELD_0.20.pickle' # 'bands_spin_EFIELD_0.20.pickle' ##'bands_spin_2D_30x30.pickle' #'bands_spin.pickle' ###
    calc_2D           = True   # are you fitting a 2D band structure?
    fin_1D_name       = 'bands_spin.pickle' #'bands_spin_CrSTe_U2.10_M001_E0.00.pickle' #'bands_spin_EFIELD_0.20.pickle'  # !! if you want to fit 2D data, but plot 1D (from the same Wannier model) to compare visually band structure 
    point_group       = 'auto_model_3mprime_p' #'auto_model_min3mprime_p' #'-3m_prime_p_type' #'C3h_empirical_2nd_and_4th' #'plain'
    explicit_E0_mstar_J0 = False
    initial_plot_only = False
    band_numbers      = [12,13,14,15]  # indexed from 1;  [12-15] for Wannier, [22-25] for DFT !!
    selection_by_energy  = True
    energy_window     = [-0.05, 0.05] #[-0.2, 0.2] #[-0.2, 0.5] #[0.0, 1.0] #[0.5, 1.5] #[1.0, 2.5] #[1.0, 2.5] #[1.0, 2.5] #[1.0, 2.5] #[-0.05, 0.05] #[-0.2, 0.2] #[-0.2, 0.5] #[0.0, 1.0] #[0.5, 1.5] #[1.0, 2.5] #[0.5, 1.5] #[1.0, 2.5] #[0.5, 1.5] #[1.0, 2.5] #[0.5, 1.5] #[1.0, 2.5] #[0.5, 1.5] #[1.0, 2.5] #[0.5, 1.5] #[1.0, 2.5] #[0.0, 1.0] #[0.5, 1.5] #[1.0, 2.5] #[-0.05, 0.05] #[-0.2, 0.2] #[-0.2, 0.5] #[0.0, 1.0] #[0.5, 1.5] #[1.0, 2.5] #[1.0, 2.5] #[1.0, 2.5] #[1, 2.5] #[-0.2, 0.5] #[-0.2, 0.5] #[-0.1, 2.5] #[-0.1, 0.1] #[-0.1, 0.1] #[-0.1, 1.0] #[1.0, 2.5] #[1.0, 2.5] #"[-0.05, 0.05] #[-0.1, 1] #[-0.1, 3] #
    weight_RMSE_spin  = 0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.2 #0.1 #1 #10000  # weight of RMSE_spin relative to RMSE_Energy
    weight_RMSE_spin_inplane = 10000 #100 #10 #10 #1 #0.1 #0.1 #0.1 #0.1 #10000 #100 #10 #10 #1 #0.1 #1 #0.1 #1 #0.1 #1 #0.1 #1 #0.1 #1 #0.1 #10 #1 #0.1 #10000 #100 #10 #10 #1 #0.1 #0.1 #0.1 #0.1 #10000 #1 #0.1 #0.1 #10000 #1 #0.1 #0.1 #0.1 #1E4
    E_thr_2D_plot     = 0.015 # eV; threshold to plot 2D spin texture around selected energy (probably Fermi)
    quiver_scale = 1
    tol = None #1e-1
    importance_of_params_refit_at_each = False  # when calculating the importance of parameters, refit at each zeroed parameter in addition to the no-refitted estimation
    exclude_params_below = 0.001
    maxiter = 50000 # 50000 # 10000 # 10000 # 10000 # 10000 # 10000 # 10000 # 10000 # 10000 # 50000 # 50000 # 10000 # 10000 # 10000 # 10000 # 10000 # 10000 # 10000 # 10000 # 10000 # 10000 # 10000 # 10000 # 10000 # 10000 # 10000 # 10000 # 10000 # 10000 # 50000 # 50000 # 10000 # 10000 # 10000 # 10000 # 10000 # 10000 # 10000 # 10000 # 1000 # 50000 # 1000
    estimate_importance = False

    init_E0       = 1.828689
    init_m_star   = -1
    init_J0       = -0.05

    # identify which parameters correspond to the usual energy offset, effective mass and exchange splitting
 #[init_E0, 0, 0, init_J0, 0, C_kinetic/(2*init_m_star)] + [0]*77
    init_params_Rashba = [1.7239525214063747, 0.28857294465555783, -0.07824787353117824, 0.27432561922128573, -8.034099493453193e-05, 0.8778606446391768, 0.0002509971215643651, -0.00043129400639264404, -0.00045204243457089895, 0.4506497503767759, -9.462797357957715, -2.7961354785603287, 0.2980366045135421, -0.08012359047264556, -0.05946553511896846, 0.008260658537356795, 0.727617016910139, 1.6352102223539013, 3.308356171126281, -0.8914566294001851, -0.25730055683625985, -0.08823660941607725, -2.220787287827502, 0.009713353059663019, 0.0019556649419074385, 0.05697470663552695, -0.00015403903219641352, -0.25581374595946016, -0.004519079119782322, 0.054788778095569396, -0.0034296946177988057, 4.397517736854358, 0.3977920958878598, 0.11955977329159273, -0.7043605172998197, 0.19948500112953138, -0.035168306185755865, -0.8149083319781294, 0.5712755414471506, 3.1526947628493893, 0.08287859524326735, -4.209346983349164, -1.2708238628958197, -0.007967754992773288, -0.0016351376757746299, 1.2163967843520282, -1.2578563748860052, -0.2686384652560584, -0.001067981595403997, -0.07679079077262191, 0.00898540252756317, 2.671437195341594e-06, -0.01583614078062137, 0.49844684031765585, 1.3208670077339995, 1.006479482389092, -0.8854308801745652, -0.12628375722798635, -0.049467039973568855, -0.04127048971995817, 0.012774662272548018, -0.21532241886691494, -5.451366716536715, 2.1895399769578674, -3.1004657009055894, 0.1485101166932582, -0.08841645620867639, -0.468290221527367, -0.6664794720510852, -0.06626991370462207, -4.747172871064235, 11.141291558930632, 2.3778021047261966, 1.3017516736061792, -0.03912700684120626, -0.42093433827311655, 0.04722109928706786, -0.013427310944467544, -4.351010714697724, 0.6434682905056794, 0.13374907470236097, -0.18382172638673622, 0.006665335319551505] #[1.8014539243564003, 0.2633791784974397, -0.01992992929775965, 0.1510601483642685, -6.360600360229727e-05, 0.5089481317078342, -0.00047589270160793354, -5.81464246800205e-05, -0.0007270683339439915, -0.5366838120670645, -9.570477429442828, -2.4209307384598686, -0.29945568787565763, -0.05309342732353034, 0.015558289970508319, -0.004867346692045081, 0.39139116162017784, 1.854538135329694, 3.5907426441100805, -0.31132587806554546, 0.07925679631756363, -0.02398019238134816, -0.9919791026702058, -0.2850607216014333, 0.0003309875661662734, -0.010436997744730789, 0.00036539022090613176, 0.04834992320172983, -0.002225015546708474, 0.030725308249938688, 0.0004820281307450052, 4.168071979022504, 0.03748823804144552, -0.06017386606518606, -0.3841290750253027, -0.08708432482505792, -0.004937983411874593, -0.7883369001422464, 0.3392101317817465, 2.688452637591328, 0.007523969896290262, -3.8020686904431367, -0.459169219874102, -0.00381883322459952, 0.000974826993288854, 1.1942269628454518, -0.9776391995874374, -0.07014186074426512, -0.0005633457531779778, -0.013434231419594302, 0.001998990173074132, -1.5655142350375377e-06, 0.009304391519675666, 0.07601880237252703, 2.0650881268476002, 0.5753980962608359, -0.20887015733305975, 0.04835753505695523, 0.25394443333214106, 0.0106424906850191, -0.008073749872800748, 0.2347923357566628, -11.437299788619857, 1.152016996419869, -0.46917519590372414, 0.10034351162259825, -0.025089423737685683, 0.11622251729871672, 0.6340398354998165, 0.01821177342532274, 1.1762831275774035, 10.652817402117348, -0.6692774050016453, 0.5228882342381167, 0.060484279212049866, 1.460284944181005, -0.025671916159502668, 0.029412597450319024, -4.190003141301428, 0.18905771913024774, 0.24957671395451442, -0.021486642726743728, 0.0013078562709273656] #[1.7086943289459486, 0.1580768933546874, -0.011508418741420876, 0.03258443436318489, 0.0003170710677922157, 0.307600487678576, 0.0006074260704694362, -0.00033773525288982706, -0.0003403322396627762, -0.5083369049559787, -9.456967014534754, -2.026173872644972, 0.14470231535802155, -0.021224331577406434, 0.025490921781845907, -0.003013762414041065, 0.2036352532216381, 2.367891987303319, 3.4818373077358915, -1.1800960455241607, 0.01729143585784588, -0.01890130204877071, -0.9388740117190597, 0.1190532241364697, 6.047837996383271e-05, -0.0038500191425906473, 0.0009017082554653676, 0.045746868751129266, 0.0040253459684940675, 0.03959714815782233, -0.00012482355744415256, 4.118456641199398, 0.009261510624915594, -0.039769030117315, -0.30229661879081093, -0.08670849717806851, -0.0022299294487690188, -1.083735439488552, 0.4804745046540162, 2.7470648200877905, -0.027760880701272785, -5.085342810183658, -0.2711815298758664, -0.601192710150727, 0.10965866604335985, 1.1663568367771922, -0.9549368322073368, -0.052567785509587, 0.06790804133534907, -0.16480873910026514, 0.0005767258635286008, -3.894404836623245e-06, 0.1968308133474107, 0.08265019033115142, 1.846267388884861, 0.6122064250823216, -0.6741598764021597, 0.021759433722531238, -0.14291967001318678, 0.005175145290009639, -0.020067040025675907, -0.11616371326190203, -11.351812803648443, 0.809247890980594, -0.5113434851771058, 0.026528158882671363, -0.00924393804538735, 0.044867070653250435, 0.6820994310120743, -0.006873755378753316, 1.061822100218491, 10.73883057142966, -0.027106053204988226, 0.6374608119705703, 0.11383801291581316, 2.4526602061211715, 0.017312006816589808, 0.13850284931324813, -4.815943721318917, -0.3519747780387475, 0.5919529498631331, 0.22238256609668347, 0.001195790025249932] #[1.5304171942749933, 0.1473896730682569, -0.006736079141853417, 0.025418374653440975, -0.004034765830358965, 0.38538481226451027, 0.0014821992027631557, 0.001170575356024718, -0.0005084228632878294, -0.41293883923910735, -8.360067350810475, -2.091357906586138, 0.12640468900508284, -0.02435719293380273, 0.024791721320712473, -0.002485181243010378, 0.23755140268576802, 2.326700743190502, 3.4639769278402888, -0.9319209183054942, 0.02064279289373882, -0.03861721379909201, -0.8912383448533672, 0.06623819868862649, -0.0002846467304053259, -0.014547197187281764, 0.00028184763242876235, 0.02574391474461874, 0.00419540039696558, 0.02243767717528196, -0.00013468820932922637, 2.85880392011984, 0.009459017912350758, -0.03825148198591086, -0.4336714510593407, -0.09488195332235316, -0.0020414368891849, -1.0315115018434886, 0.4632770878363526, 2.5659350627604747, -0.02704299098844691, -8.623622446643274, -0.18344734856664502, -0.3489223555982144, 0.15457184334935836, 1.0411599453506755, -0.5112166444307056, -0.05393074636231618, 0.07263532001591566, -0.49390128897612584, 0.0002523146232156126, -3.08045836777077e-05, 0.20533904721533325, 0.08216695888413769, 0.9358792133958254, 1.253907854702204, -1.2784824273119906, 0.02746799689255497, -0.14209467058691655, 0.0037615462353295503, -0.03217934638978437, -0.10144519050896444, -10.922309543581065, 0.6179223609770084, 1.3595255064207699, 0.019683489608049498, -0.07275526427381249, -0.5908224688531551, 0.5232743499199852, -0.009766954586836928, 0.4484862437320566, 11.167735564191666, 2.505489910724374, 0.7149980028141902, 0.09060875777549476, 1.747321041040455, 0.018384464955928576, 0.09631214183004205, -6.8247714336123195, -0.2620975143227573, 0.6835235254096768, 0.23372922906701143, 0.0006809402530110983] #[1.6185323633367261, 0.09328873866459983, -0.00704506644223827, 0.021104212417190397, -0.004872325132959959, 0.7801355486290145, -0.0021092898108927925, -0.0008048374506515821, 0.000958787246396443, -0.4286506653418317, -8.722621917382813, -1.3775440096033842, 0.35607035857332925, -0.10885805092535514, 0.006213047111766049, -0.0010534692857455683, 0.227470177385445, 1.9559469394511795, 3.4038222175551907, -0.3511282974241404, 0.005664345485849943, -0.03229364614307711, -1.5245209450752668, 0.047167112670287836, -0.00010714649233117338, 0.03482937704301952, 5.071273425464566e-05, 0.04242833324319879, 0.002614990725265852, -0.003915491908186863, 0.00020395202513992137, 1.8586937050647365, 0.007155048645044211, -0.0275050683947259, -0.2428301861858167, -0.37406563285860295, -0.5648302912511216, -1.055064826657772, 0.21922274091992328, 2.737631131795129, -0.005868535447837962, -12.329784841975872, -1.051235423518349, -0.4801005610692759, 0.6549771490793096, 0.71183936470764, -0.6031435109161831, -0.04656318707073394, 0.10961280118325634, -0.8932577028033035, -0.00011536071697335957, 7.326653718389041e-05, 0.11422296709518864, 0.05716514927832039, 0.39987955151809906, 2.7502280614749077, -1.1975449036273265, 0.3337945850327543, 0.0984574815812086, -0.005518537384097841, -0.01521766448290774, -0.06941957098222148, -4.361992182171544, 0.42156179323202225, 1.8940391765484885, 0.027455418832165006, -0.21596567388075683, -1.3156427237950468, 0.26068394454262966, -0.007779969717956553, -6.747016322047649, 8.042971686580932, 1.1919939630891294, 0.40425737053257405, 0.04870704882280834, 1.955602604089711, 0.029795200654743267, 0.16827013891391596, -13.759534127358766, -0.9069126553779872, 0.45534525560963013, 0.21266084020388565, 0.00044409023254009727] #[1.5506771987500692, 0.23923516869289607, -0.008354014549080128, 0.24121948850258823, -0.002859319198926801, 0.6417337034638706, -0.06328176762353607, -0.0004376800295802217, 0.0006848502067525143, 0.3088907053219121, -8.716888629154766, -2.4454981264837947, 0.4495193005807121, -0.03579019740303954, 0.07145203997356264, -0.0005348685049202719, 0.1613827312266547, 1.6949740124127395, 1.8720897698894445, -0.3456456852539007, 0.004252691790575951, -0.01377851039112897, -2.4550812146449754, 0.028511529544380285, -0.000319471204466139, -0.12365555749219503, 0.0005457224801503967, 0.03143609556365795, 0.003389906848474752, -0.0022930913182636348, -0.0023715762053875405, 1.9585433170670736, 0.01745105899368276, -0.04425464132303056, -1.1576859311904228, -0.22318139867569142, -0.4440096721424325, -0.5582878665164186, -0.13972851660773788, 8.387012112236349, -0.024719984457488006, -4.385874111584087, -0.641757636329543, -0.4302691680337978, 0.5752373413076728, 0.7323916223445806, -0.4343506398285055, -0.07952086645095538, -0.2479802968708147, -0.3497130291382735, -0.005359662865680324, 4.763345502145893e-05, 0.18747030235566708, 0.05015563940962947, 1.055591469528919, 0.7768290076236912, -1.3132114971876074, 0.2131282619237317, 0.07876966998039331, -0.003089370500600234, -0.03911463277299773, -0.15104713241003975, -2.8978986590484475, 0.19832476944430966, 1.9157033504003165, 0.037465558797747545, -0.1438041842221942, -1.0811495859640594, 0.45786643470238775, -0.007230820400896735, -5.803219634941982, 4.844850070066736, 1.3178011475704254, 0.7278615991682527, 0.058244687236256726, 1.1487476577614775, 0.06277890181865428, -1.5065443572128983, -6.454854545140208, -0.57395201654417, 0.2992256364852579, 0.6741731135013902, 0.0003917892006878802] #1.697416601863445, 0.2662482350081701, -0.008349334779739766, 0.11026042381986935, -0.053326116337001024, 0, 0, 0, 0, 0, -9.4510540868784, -0.5644960550657914, -2.0032787730758876, -0.37517583664667803, 0.28204597837205303, -1.0218962780212193, -0.5798528101518049, 0.8848164407501029, 3.2220280886801076, 2.1329091435668532] + [0]*63 #1.697416601863447, 0.2662482350081701, -0.008349334779739766, 0.11026042381986935, -0.053326116337001024, 0, 0, 0, 0, 0, -9.4510540868784, -0.5644960550657914, -2.0032787730758876, -0.37517583664667803, 0.28204597837205303, -1.0218962780212193, -0.5798528101518049, 0.8848164407501029, 3.2220280886801076, 2.1329091435668532] + [0]*63 #1.697416601863447, 0.2662482350081701, -0.008349334779739766, 0.11026042381986935, -0.053326116337001024, 0, 0, 0, 0, 0, -9.4510540868784, -0.5644960550657914, -2.0032787730758876, -0.37517583664667803, 0.28204597837205303, -1.0218962780212193, -0.5798528101518049, 0.8848164407501029, 3.2220280886801076, 2.1329091435668532] + [0]*63 #1.697416601863447, 0.2662482350081701, -0.008349334779739766, 0.11026042381986935, -0.053326116337001024, 0, 0, 0, 0, 0, -9.4510540868784, -0.5644960550657914, -2.0032787730758876, -0.37517583664667803, 0.28204597837205303, -1.0218962780212193, -0.5798528101518049, 0.8848164407501029, 3.2220280886801076, 2.1329091435668532] + [0]*63 #[1.5679412763299, 0.14396131790112704, -0.0020825414405256336, 0.17022031586602948, -0.009458195642342994, 0.2002071401442831, 0.10907242264504553, -0.0005451363069502039, 0.00015237313169328868, -0.1313474746421388, -8.981003923110734, -2.391973650646041, -0.16475191897831248, -0.03464155243063036, 0.10605305117401419, 0.0020781297174344185, 0.12351937009963795, 1.9362928079756512, 2.098372574801844, 0.10546173318407889, -0.00030004916120513636, -0.004939588856634042, -0.5364705036431239, 0.011376535301548907, 6.0254692659768466e-05, -0.4456143436763724, 0.0004836960683825076, 0.016333873170968086, 0.001530981063445451, -0.0007650924206274303, -0.0008418324633106325, 0.8686543758358741, 0.060827464114568415, -0.08261436764843338, 0.9034145062452308, 0.26552124448855263, -0.47471699353333735, -0.5629422969196585, -0.28559799221808946, 8.63049979984653, -0.005310791309710821, -5.423351840438334, -0.6483714456936094, -0.5002025909339394, 1.0978894733479112, 0.9430696013257658, -0.17279721705985573, 0.2140697781705892, 0.5525508345787393, -0.18905063623148255, -0.0009266733803338412, -2.9114343972039646e-05, -0.10851894689241189, -0.26106795959419504, 0.4069754200420648, 0.11092150245105879, -0.6069290143464403, 0.16784994584321805, 0.04207546391433871, -0.0006711268640967424, 0.02350241350251333, -0.03170573304838531, -0.702200763847525, 0.05124229029546043, 0.6177657553617409, -0.06987001207197939, 0.08295296545232446, 0.26409834033537416, 0.9379788909079183, 0.20458753678866953, -4.424725970211837, 3.1893474184376283, 0.3496088729495994, 0.47312890737415497, 0.04708690886412042, 1.468479796276982, -0.037955424050272454, -2.2507300272974318, -5.80871218283851, -1.0189116212095972, 0.27628583879620305, 0.4115960473497522, 0.00027299591513668453] #[1.6463304370641543, 0.03401302633943676, 0.0015932468973529994, 0.0445449791795894, -0.004795399760301956, -0.02608134192294515, 0.25381220970371515, 0.0005778801211781375, 0.0002155778542553821, -0.060054862508304475, -9.27090666966994, -2.2113064313298856, -0.07369024343764385, 0.027330364227026293, 0.04558936846997354, 0.25043235372768907, 0.3279990924532617, 2.1072737157121284, 2.5120745574131056, 0.23604785622067376, -0.00017850448577761132, 0.001655637967301063, -1.0252045939704182, -0.01914011297282246, 0.0009260564716569206, -0.05162911718916425, 0.0001670345661185343, 0.005242808231017786, 0.0002981827167736727, -0.00029793232425158743, -0.00043677892258556423, 0.9408896594655433, -0.16138902804118077, 1.0243782541628876, 0.5138419567507374, 0.5031162177984467, 0.1612109500562903, -0.979053367470178, 0.11542944459535717, 7.137798450908383, -0.0012241203488571552, -2.391836590276253, -0.737406913059411, 0.530960908344642, 4.201105853784177, -0.32177378573951976, -0.08035739287495162, 0.16313496948452239, 0.35946295201054845, -0.04948776389264749, 0.02156794313778203, 8.258319434661503e-06, -0.012237850483866426, -0.07509794047719844, 0.3250692398305444, -0.03173757970268187, -0.3010671236353622, -0.08522547462887717, 0.11698291990700582, -0.0002503736182903405, -0.05468226202016375, -0.017697541281035768, -0.8458407404303908, 0.010821130805351247, -0.6054599436668028, -0.1810970023523552, -0.04444596711275379, 0.07279081965156638, 0.7598911750895738, 0.10811786740377635, -1.8741152005255504, 1.208420574420854, 0.23660964523632577, 0.0815233107726331, -1.3956928627296619, 2.523021038897373, 0.04080689361747794, 1.647064772943142, -3.491561017215162, 0.5565993687014391, 0.23533539312622925, -0.29729301614640946, 0.00024121451310470403] #[1.6910208725045348, 0.031553072721924924, 0.0011709390357725145, 0.032212108024177885, -0.0033601836054867806, -0.014435858825661441, 0.07640905386098301, 0.000831692453017163, 0.00030756852412668415, -0.059318573483196996, -8.910025579860548, -3.348202141644303, -1.2019809062728974, 0.05686486857070461, 0.051632879536182036, 0.17471256319006, 0.3444007655819954, 2.204307845075377, 1.4200870961949992, 0.13905944468862969, -0.0001677031229082088, 0.0019022881978346135, -1.756026789576877, -0.021500178142126708, 0.0010123614899656143, -0.05194164080810407, 0.00025506748295976744, 0.004254141077258905, 0.00021348575546044075, -0.0001586847411757625, -0.0009086462749898325, 0.8827349849272306, -0.15013321358260073, 1.6785245586483684, 0.6145377378064707, 0.19494784676266919, 0.07382103758823652, -0.7298712349057208, 0.12928719850374876, 8.618477796100493, -0.0012792422387108374, -2.261348834497908, -0.7166695865195494, 0.48751754802907477, 4.049684636457753, -0.21253902429828242, -0.1122476613402261, 0.18522117025044593, 0.612913772765721, -0.2635482881521303, 0.07918879916514479, 8.067048766548358e-06, -0.01363433904861211, -0.06117787248212481, 0.15770702736939402, -0.02239064357208407, -0.28146986414246833, 0.3930128152860959, 0.0779837410382862, -0.00031218345548834404, -0.11371209429241161, -0.015730299110848688, -0.7936232707268509, 0.006882913937927754, -0.7056411201729902, -0.1589940878938887, -0.03563587551231992, 0.07372636398245686, 0.7380869256833553, 0.11730781109421536, -2.500679097487888, 1.1297761706663199, 0.21047135633267947, 0.07635611164714817, 0.66062124142548, 5.299880521041265, 2.176964488935134, 10.89276530177559, -9.643890473477203, 0.5358094151870352, 0.1976956060676649, -0.2129792292070044, 0.00021447402583248484] #[1.3937994437932395, 0.040454533540395216, 0.014646970628485922, 0.027474212047059444, -0.0025221955849410727, -0.009039935286487666, 0.034562650052938904, 0.0008795224764419295, 0.0019025374448353683, -0.05353379844626227, -7.092780311148664, -1.8034452494169013, -0.5233972851221151, 0.07711186944787282, 0.03982274424623142, 0.11369128981393117, 0.1806734504018267, 1.7596213434716752, 3.0203537539549874, 0.3323353578838999, -0.00012886655297109572, -0.028619435149307808, -0.6983748278275614, -0.029842613927660314, 0.000686988779636348, -0.037951886776178625, 0.0002301719819057776, 0.002682825640890965, 0.00021846513178963524, -0.0006289104606673828, -0.0007544791907477348, 0.3734837633702818, 0.22519337133791717, 1.1199300291286711, 0.589784745199222, 0.14757628581486815, 0.19345444649846788, -0.6999196086092261, 0.17904777293928736, 7.3726413304277045, -0.010975633419174832, -2.24674621016527, -0.1589579249155877, 0.6112962714736481, 1.9123979493858814, -0.1463206585551015, -0.09025207063009513, 0.18681917648598761, 0.7481369110267844, -0.20632688346957578, 0.05842081677602484, 6.349695700909319e-06, -0.010646805162438377, -0.04177102964022382, 0.08249087697745516, -0.029910967499895594, -0.2719199247361833, 0.09540189350184367, 0.23028635931871905, 0.003002790093653341, -0.10644914328226396, -0.009189103099676937, 1.036953486470121, 3.7976976606609583, 8.361135345705677, -0.21260775713202368, -0.025706344650447868, -0.8933798498585832, 0.68864993385038, 7.753308931820786, -25.86903640437106, -10.547196717150744, -1.2641539849680892, 0.09452245595842396, -0.7843239957980808, 5.145819743736384, 2.1814740966789747, 12.392617244753175, -10.358109645327715, 0.8578430961199808, 0.9812023495440183, -0.1555101266976216, -0.00045001717630524193] #[1.5743730692561764, 0.020661593644269616, 0.004495089815535389, 0.017971146511584248, -0.0025192481114781062, -0.0172109646256778, 0.027660279262352705, 0.004666515610510383, 0.010783200829158339, 0.03030085204355345, -8.201496112748393, -1.3561272342241844, -0.2977520646631747, 0.0873957012962789, 0.04621563323359755, 0.06824517039341968, 0.11335837069459445, 1.6889883450837986, 3.886401276335218, 0.4313098477644558, -0.0002650179077819699, -0.018199716457802016, -0.6491158055429791, -0.023570608082961554, 0.04818044849986847, -0.020312321744707, 0.0002300511808943796, 0.002060371821662353, 0.0002171590902358297, -0.010960954797454002, -0.0012391430880650345, 0.18974101239641195, 0.4235392459210544, 4.5531885523902815, 0.9742523127661504, 0.4491674172637321, 0.14060691889690163, -0.8006060737549567, 0.41094111280143975, 5.104012316862046, -0.053728354385199364, -0.7618577517489058, -0.8235830541834888, -1.6673171760038539, 3.3485961142143212, -0.3144361459964622, -0.08617456968610265, 0.04266611241424341, 0.325046699806545, -0.16056029936614233, 0.051523625554108374, 0.08678396971025312, -0.00504069427855898, -0.05227151782626736, 0.13338401884134354, -0.06013365732831165, -0.19641737254802494, 0.06912981959103295, 0.10439936955715717, 0.0025205751732035983, -0.172056443252947, 0.0312726385817423, 1.2849998636626392, 18.890681925108154, -17.056680305174986, -0.17930843019990567, -0.03183899388680747, -0.6319628980365917, 0.9764065530928021, 2.641510327595398, -42.666450076663864, -12.266696211248124, -2.7165378985643587, 0.07160598428862869, -0.8671705109376027, 7.721616790156608, 2.2840098635990396, 8.825100689893834, -7.449033276163915, 0.5437164858540088, 2.160369549683309, -0.11118228231354707, -0.003286597861108867] #[0.5416445871150769, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757] #1.697416601863445, 0.2662482350081701, -0.008349334779739766, 0.11026042381986935, -0.053326116337001024, 0, 0, 0, 0, 0, -9.4510540868784, -0.5644960550657914, -2.0032787730758876, -0.37517583664667803, 0.28204597837205303, -1.0218962780212193, -0.5798528101518049, 0.8848164407501029, 3.2220280886801076, 2.1329091435668532] + [0]*63 #[0.5416445871150769, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757] #1.697416601863445, 0.2662482350081701, -0.008349334779739766, 0.11026042381986935, -0.053326116337001024, 0, 0, 0, 0, 0, -9.4510540868784, -0.5644960550657914, -2.0032787730758876, -0.37517583664667803, 0.28204597837205303, -1.0218962780212193, -0.5798528101518049, 0.8848164407501029, 3.2220280886801076, 2.1329091435668532] + [0]*63 #[0.5416445871150769, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757] #1.697416601863445, 0.2662482350081701, -0.008349334779739766, 0.11026042381986935, -0.053326116337001024, 0, 0, 0, 0, 0, -9.4510540868784, -0.5644960550657914, -2.0032787730758876, -0.37517583664667803, 0.28204597837205303, -1.0218962780212193, -0.5798528101518049, 0.8848164407501029, 3.2220280886801076, 2.1329091435668532] + [0]*63 #[0.5416445871150769, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757] #1.697416601863445, 0.2662482350081701, -0.008349334779739766, 0.11026042381986935, -0.053326116337001024, 0, 0, 0, 0, 0, -9.4510540868784, -0.5644960550657914, -2.0032787730758876, -0.37517583664667803, 0.28204597837205303, -1.0218962780212193, -0.5798528101518049, 0.8848164407501029, 3.2220280886801076, 2.1329091435668532] + [0]*63 #[0.5416445871150769, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757] #1.697416601863445, 0.2662482350081701, -0.008349334779739766, 0.11026042381986935, -0.053326116337001024, 0, 0, 0, 0, 0, -9.4510540868784, -0.5644960550657914, -2.0032787730758876, -0.37517583664667803, 0.28204597837205303, -1.0218962780212193, -0.5798528101518049, 0.8848164407501029, 3.2220280886801076, 2.1329091435668532] + [0]*63 #[0.5416445871150769, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757] #1.697416601863445, 0.2662482350081701, -0.008349334779739766, 0.11026042381986935, -0.053326116337001024, 0, 0, 0, 0, 0, -9.4510540868784, -0.5644960550657914, -2.0032787730758876, -0.37517583664667803, 0.28204597837205303, -1.0218962780212193, -0.5798528101518049, 0.8848164407501029, 3.2220280886801076, 2.1329091435668532] + [0]*63 #[0.5416445871150769, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757] #[0.5416445871150766, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757] #[0.5416445871150766, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757] #[0.5416445871150766, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757] #[0.5416445871150766, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757] #[0.5416445871150766, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757] #1.697416601863445, 0.2662482350081701, -0.008349334779739766, 0.11026042381986935, -0.053326116337001024, 0, 0, 0, 0, 0, -9.4510540868784, -0.5644960550657914, -2.0032787730758876, -0.37517583664667803, 0.28204597837205303, -1.0218962780212193, -0.5798528101518049, 0.8848164407501029, 3.2220280886801076, 2.1329091435668532] + [0]*63 #[0.5416445871150766, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757]
 #[0.5416445871150766, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757]
 #[0.5416445871150766, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757]
 #[0.5416445871150766, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757]
 #[0.5416445871150766, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757]
 #[0.5416445871150766, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757]
 #1.697416601863445, 0.2662482350081701, -0.008349334779739766, 0.11026042381986935, -0.053326116337001024, 0, 0, 0, 0, 0, -9.4510540868784, -0.5644960550657914, -2.0032787730758876, -0.37517583664667803, 0.28204597837205303, -1.0218962780212193, -0.5798528101518049, 0.8848164407501029, 3.2220280886801076, 2.1329091435668532] + [0]*63 #1.697416601863447, 0.2662482350081701, -0.008349334779739766, 0.11026042381986935, -0.053326116337001024, 0, 0, 0, 0, 0, -9.4510540868784, -0.5644960550657914, -2.0032787730758876, -0.37517583664667803, 0.28204597837205303, -1.0218962780212193, -0.5798528101518049, 0.8848164407501029, 3.2220280886801076, 2.1329091435668532] + [0]*63 #[1.697416601863445, 0.2662482350081701, -0.008349334779739766, 0.11026042381986935, -0.053326116337001024, 0, 0, 0, 0, 0, -9.4510540868784, -0.5644960550657914, -2.0032787730758876, -0.37517583664667803, 0.28204597837205303, -1.0218962780212193, -0.5798528101518049, 0.8848164407501029, 3.2220280886801076, 2.1329091435668532] + [0]*63
#[0.5416445871150766, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757]
 #[0.5416445871150766, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757]
 #[0.5416445871150766, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757]
 #[0.5416445871150766, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757]
 #[0.5416445871150766, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757]
 #[0.5416445871150766, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757]
 #[0.5416445871150766, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757]
 #[0.5416445871150766, 0.2380229639984599, 0.00150584052707465, 0.031405862250710403, -0.0022647327911686325, 0.08791105003640132, 0.021041624744052287, 0.0033228567536326206, 0.007884749608662974, 0.029593374589318933, -2.7010636433560116, -2.1075784826720465, -0.22635782821625638, 0.09759083562162019, 0.0549939045340779, 0.16981146708498474, -0.05155192241010503, 0.9058014765112259, 1.3989749466498882, 0.22269202159274126, -0.00020593163823738776, 0.031330086148377456, -0.3802298828324273, 0.07241225086787487, 0.008305923683183475, -0.03138096837759331, 0.010590486563731571, -0.006973738007328867, 0.000700874374245976, 0.01869142701528447, 0.0054571441950546015, -0.3417376485795609, -0.20004369754063356, -2.771814541296804, 0.34321336236141997, -0.12489958419441227, 0.11907280689637684, -0.30784326559192077, 0.1842886503795741, 3.242750960575062, -0.02516555501871988, -0.3183690101475747, -0.4182694922954544, 0.3691615878659269, 1.019610855317067, 0.9435122228596962, 0.12266878873150627, 0.03138893016645607, 0.14222012010236262, -0.09793666001571372, -0.16381779863183654, 0.04772163034909288, 0.039586649485311834, -0.19449542382134088, 0.08998844655354771, -0.048847630230771766, -0.09121236902462249, 0.11340209303276494, 0.05961949900536603, -0.013134760985239892, -0.10558239387877849, 0.022064124753659847, 0.5266793988685905, 8.267132773167663, 10.136079968216102, -0.22478648529392187, -0.0277030609444731, -1.011316915709536, 2.3882716984380297, 1.1129684998777059, -12.373317271528599, -5.715170703656359, -2.095761037671469, 0.751198739720162, -0.3336176051755091, 2.82547509726915, 2.690772584607579, 3.5968246271671873, -4.597010468125971, 2.553895321953364, 1.508273656338945, -0.10890863557790903, 0.25138155301065757]
 #[1.697416601863445, 0.2662482350081701, -0.008349334779739766, 0.11026042381986935, -0.053326116337001024, 0, 0, 0, 0, 0, -9.4510540868784, -0.5644960550657914, -2.0032787730758876, -0.37517583664667803, 0.28204597837205303, -1.0218962780212193, -0.5798528101518049, 0.8848164407501029, 3.2220280886801076, 2.1329091435668532] + [0]*63
# [0]*83 #[1.4055549341287061, 0.333377577817121, -0.009974643834194061, -0.351150622046644, -0.8463333536499065, -7.734714297601369, -2.430539833779818, -2.2753773198028675, -0.06872879521889111, -0.19996187254137676, -0.13958591664237957, 0.14155773116346027, 3.7501463265612527, 3.200581941755275, 1.6315778751816985] + [0]*68
# [1.4055549341287061, 0.333377577817121, -0.009974643834194061, -0.351150622046644, -0.8463333536499065, -7.734714297601369, -2.430539833779818, -2.2753773198028675, -0.06872879521889111, -0.19996187254137676, -0.13958591664237957, 0.14155773116346027, 3.7501463265612527, 3.200581941755275, 1.6315778751816985, 0.0002938832888469938, 0.27043624080546524, 0.8338614114456169, 0.09742904703403513, 0.40552818842749055, -0.1891553775168351, -0.11034411785866428, 0.5543074129090738, 1.1377571112386349, -0.3692358515691372, 8.166223978752644, -0.10968138875400407, 0.1142135544581728, 0.3895901493331322, -1.0282142716827078, 0.0336606366331121, -4.938354159444606, -1.169827290523522, -0.37381935284002454, -0.7181485385528363, -0.7362322826103672, -1.4044327795977711, 1.5814206853562789, 1.5500668791929764, -0.6727882828393088, -2.1652717222659055, 0.9401416966924768, -0.00010767782411209189, -0.09057127063073461, -0.06107118330447879, -3.9318275809894705, -4.765280932817918, 4.578502654666096, -0.5389375569574171, 0.13460456763742332, 0.304054057771897]
    # init_params_Rashba[5] = C_kinetic/(2*init_m_star)
    # bounds = [(-1e3, 1e3)]*3 + [tuple([0,0]) for i in range(24)]
    # bounds[11] = (-1e3, 1e3)
    # bounds[14] = (-1e3, 1e3)
    # bounds[17] = (-1e3, 1e3)
    bounds = None

    yaxis_lim = [-0.5, 2.5]
    color_axis_lim_Sz = [-1, 1]
    color_axis_lim_Sxy = [-0.2, 0.2]
    shift_bands_by_fermi = True
    E_F = float(np.loadtxt('../FERMI_ENERGY.in')) #float(np.loadtxt('FERMI_ENERGY_EFIELD_0.20.in'))
    system = 'DFT'  # not useful for now
    k_lim = 0.6
    fit_options = {'disp':True, 'maxiter':maxiter} #50000}

    hyperparameters = [weight_RMSE_spin, weight_RMSE_spin_inplane, band_numbers, point_group, selection_by_energy, energy_window, tol, exclude_params_below, fin_name, fin_1D_name, calc_2D, E_thr_2D_plot]
    hyperparameter_names = ['weight_RMSE_spin', 'weight_RMSE_spin_inplane', 'band_numbers', 'point_group', 'selection_by_energy', 'energy_window', 'tol', 'exclude_params_below',  'fin_name', 'fin_1D_name', 'calc_2D', 'E_thr_2D_plot']
    # ================================================================================
    # ================================================================================

    init_params = [init_E0, init_m_star, init_J0, *init_params_Rashba] if explicit_E0_mstar_J0 else init_params_Rashba
    fitter = RashbaFitter()
    if shift_bands_by_fermi is True:
        fitter.set_Fermi(E_F)
    fitter.set_calc_2D(calc_2D)
    fitter.explicit_E0_mstar_J0 = explicit_E0_mstar_J0
    fitter.initial_plot_only = initial_plot_only

    fitter.save_hyperparameters(hyperparameters, hyperparameter_names)
    fitter.load_spin_texture(fin_name, band_numbers=band_numbers, energy_lim=energy_window, \
                                selection_by_energy=selection_by_energy, \
                                shift_by_fermi=shift_bands_by_fermi, \
                                k_lim=k_lim, kx_lims=[-k_lim, k_lim], ky_lims=[-k_lim, k_lim]
                            )
    # fitter.define_model_Hamiltonian(fun=lambda k_vec, E0, m_star, J0, S_vec, params: model_H(k_vec, E0, m_star, J0, S_vec, params, point_group=point_group))
    fitter.define_model_Hamiltonian(model_3mprime_p)
    fitter.fit_model_to_bands(init_params, system=system, tol=tol, bounds=bounds)
    if estimate_importance:
        fitter.importance_of_parameters(refit_at_each=importance_of_params_refit_at_each, exclude_params_below=exclude_params_below)
    print(fitter.print_fit_results())
    fitter.save_fit_results()
    fitter.plot_fit(fitter.best_params, E_F=fitter.E_F, E_thr_2D_plot=E_thr_2D_plot, quiver_scale=quiver_scale, fin_1D=fin_1D_name, \
                    shift_by_fermi=shift_bands_by_fermi, clim_Sz=color_axis_lim_Sz, clim_Sxy=color_axis_lim_Sxy)
    # fitter.print_fitted_Hamiltonian()


def main():
    fit()
    
    # # for D3d_Mz to C3v_Mz
    # D3dMz_to_C3vMz = list(range(0, 5)) + list(range(10, 20)) + list(range(31, 46)) + list(range(62, 83)) 
    # params_Rashba = [0.7191776128067284, 0.2192076283423997, -0.015999513005388488, 0.020948187342127954, -0.024530508123269304, -4.222535941618069, -2.5019681623658707, -2.2765311482647155, 0.2919387589963034, 0.04961139978231685, -0.2362714581585037, 0.07446296807217856, 1.5167204329155357, 0.8473719337618446, 0.45461682490929267, -0.0997344188101407, -0.3125760257089472, -3.9745903597134467, -0.24871185370089155, 0.5491133292384869, 2.8039025240700974, -0.4514072112094158, 0.3286342558358919, 3.0467272468142905, -0.00015189101156396146, -0.09639072419568989, -0.0453990168350213, 0.17343712968693595, 0.038835860782200346, 0.10894291726204046, -0.017454551202859023, 7.149657028642984, 2.49590099635686, 0.024202345242901546, 0.6183869856696373, 4.204018818265769, -0.47083202111796574, -0.6726955558817309, -9.580876930753519, 1.7108350277106026, -0.46160649594604986, -0.31621655674620575, 0.15475832179154325, 0.5084495220867575, -0.3745196648629757, 1.4553775085537493, -2.8515484403968996, 0.7270724423770567, 0.44465624084875366, 0.25006526585991823, 1.2320488962902332]
    # print(convert_params(old_params=params_Rashba, positions_in_new_list=D3dMz_to_C3vMz, num_params_new=83))


if __name__ == '__main__':
    main()

