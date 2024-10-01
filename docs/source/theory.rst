Theory
=============

We utilize the maximally localized Wannier functions (MLWF) of Marzari and Vanderbilt [`N. Marzari and D. Vanderbilt, PRB (1997) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.56.12847>`_; `N. Marzari et al., Rev. Mod. Phys. (2012) <https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.84.1419>`_] within the ``wannier90`` package 
[`A. A. Mostofi et al., Comput. Phys. Commun. (2008) <https://www.sciencedirect.com/science/article/pii/S0010465507004936?via%3Dihub>`_; `A. A. Mostofi et al., Comput. Phys. Commun. (2014) <https://www.sciencedirect.com/science/article/pii/S001046551400157X?via%3Dihub>`_] to construct effective tight-binding representation of the *ab initio*-calculated ground state. 

Accessing the spin expectation values of the interpolated band structure with Wannier tight-binding models requires deriving the Wannier real-space spin operator.   

Just like any other operator that can be obtained from the *ab initio* calculation (for all the *ab initio* k-points :math:`\textbf{q}`), the *spin operator* in the basis of *ab initio* eigenstates :math:`\mathcal{S}_{mn}^\mathrm{H}(\textbf{q})` can be converted into its real-space representation by first applying the semi-unitary matrix :math:`V_{m^{\prime} m}(\textbf{q})` to convert :math:`\mathcal{S}` from the *Hamiltonian* gauge to the *Wannier* gauge

.. math::
    \begin{equation}
        \mathcal{S}_{m n}^{\mathrm{W}}(\textbf{q})=\sum_{m^{\prime} n^{\prime}} V_{m^{\prime} m}^\dagger(\textbf{q}) \cdot \mathcal{S}_{m^{\prime} n^{\prime}}^\mathrm{H}(\textbf{q}) \cdot V_{n^{\prime} n}(\textbf{q}),
    \end{equation}

where the spin operator :math:`\hat{\mathcal{S}}` projected onto the basis of the *ab initio* eigenstates :math:`\psi` in its matrix form :math:`\mathcal{S}_{m^{\prime} n^{\prime}}^\mathrm{H}(\textbf{q}) = \left\langle\psi_{m^{\prime} \textbf{q}}|\hat{\mathcal{S}}| \psi_{n^{\prime} \textbf{q}}\right\rangle` is obtained with the help of the ``vaspspn`` module of ``WannierBerri`` (`S. Tsirkin, npj Comp. Mater. (2021) <https://www.nature.com/articles/s41524-021-00498-5>`_). Primed indices run over the (larger) space of disentanglement bands.

Follows the direct Fourier sum over the *ab initio* grid

.. math::
    \begin{equation}
        \mathcal{S}_{m n}(\textbf{R}) \equiv \frac{1}{N_{\textbf{q}}} \sum_{\textbf{q}}  \mathcal{S}_{m n}^{\mathrm{W}}(\textbf{q}) \cdot e^{-i \textbf{q} \cdot \textbf{R}} \,,
    \end{equation}

with :math:`N_{\textbf{q}}` the number of *ab initio* grid points :math:`\textbf{q}`.


Having :math:`\mathcal{S}_{m n}(\textbf{R})` at hand, its interpolation :math:`\overline{\mathcal{S}}_{mn}^\mathrm{H} (\textbf{k})` to an *arbitrary* k-vector :math:`\textbf{k}` involves an inverse Fourier sum :math:`\overline{\mathcal{S}}_{mn}^\mathrm{W} (\textbf{k}) = \sum_\textbf{R} \mathcal{S}_{mn} (\textbf{R}) \cdot  e^{i \textbf{k} \cdot \textbf{R}}` over the real-space lattice vectors :math:`\textbf{R}` followed by a rotation back to the Hamiltonian gauge of the original eigenstates :math:`\overline{\mathcal{S}}_{mn}^\mathrm{H} (\textbf{k}) = (U^\dagger \cdot \mathcal{S}^\mathrm{W} \cdot U)_{mn}`, where :math:`U_{mn}` is a unitary matrix which diagonalizes the interpolated Hamiltonian :math:`\overline{\mathcal{H}}_{mn}^\mathrm{W} (\textbf{k}) = \sum_\textbf{R} \mathcal{H}_{mn} (\textbf{R}) \cdot  e^{i \textbf{k} \cdot \textbf{R}}`.

See the `Supplementary <https://pubs.acs.org/doi/suppl/10.1021/acs.nanolett.4c03029/suppl_file/nl4c03029_si_001.pdf>`_ of `L. Vojáček*, J. M. Dueñas* et al., Nano Letters (2024) <https://pubs.acs.org/doi/10.1021/acs.nanolett.4c03029>`_.

..  This procedure is implemented in ``spinWannier``.