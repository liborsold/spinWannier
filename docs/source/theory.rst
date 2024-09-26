Theory
=============

Accessing the spin expectation values of the interpolated band structure with Wannier tight-binding models requires deriving the Wannier real-space spin operator.   

Just like any other operator that can be obtained from the *ab initio* calculation (for all the *ab initio* k-points :math:`\textbf{q}`), the *spin operator* in the basis of *ab initio* eigenstates :math:`\mathcal{S}_{mn}^\mathrm{H}(\textbf{q})` can be converted into its real-space representation by first applying the semi-unitary matrix :math:`V_{m^{\prime} m}(\textbf{q})` to convert :math:`\mathcal{S}` from the *Hamiltonian* gauge to the *Wannier* gauge

.. math::
    \begin{equation}
        \mathcal{S}_{m n}^{\mathrm{W}}(\textbf{q})=\sum_{m^{\prime} n^{\prime}} V_{m^{\prime} m}^\dagger(\textbf{q}) \cdot \mathcal{S}_{m^{\prime} n^{\prime}}^\mathrm{H}(\textbf{q}) \cdot V_{n^{\prime} n}(\textbf{q}),
    \end{equation}

where the spin operator :math:`\hat{\mathcal{S}}` projected onto the basis of the *ab initio* eigenstates :math:`\psi` in its matrix form :math:`\mathcal{S}_{m^{\prime} n^{\prime}}^\mathrm{H}(\textbf{q}) = \left\langle\psi_{m^{\prime} \textbf{q}}|\hat{\mathcal{S}}| \psi_{n^{\prime} \textbf{q}}\right\rangle` is obtained with the help of the \texttt{vaspspn} module of \texttt{WannierBerri}~\cite{tsirkin_high_2021}. Primed indices run over the (larger) space of disentanglement bands.

Follows the direct Fourier sum over the *ab initio* grid

.. math::
    \begin{equation}
        \mathcal{S}_{m n}(\textbf{R}) \equiv \frac{1}{N_{\textbf{q}}} \sum_{\textbf{q}}  \mathcal{S}_{m n}^{\mathrm{W}}(\textbf{q}) \cdot e^{-i \textbf{q} \cdot \textbf{R}} \,,
    \end{equation}

with :math:` N_{\textbf{q}} ` the number of *ab initio* grid points :math:`\textbf{q}`.


Having :math:`\mathcal{S}_{m n}(\textbf{R})` at hand, its interpolation :math:`\overline{\mathcal{S}}_{mn}^\mathrm{H} (\textbf{k})` to an *arbitrary* k-vector :math:`\textbf{k}` involves an inverse Fourier sum :math:`\overline{\mathcal{S}}_{mn}^\mathrm{W} (\textbf{k}) = \sum_\textbf{R} \mathcal{S}_{mn} (\textbf{R}) \cdot  e^{i \textbf{k} \cdot \textbf{R}}` over the real-space lattice vectors :math:`\textbf{R}` followed by a rotation back to the Hamiltonian gauge of the original eigenstates :math:`\overline{\mathcal{S}}_{mn}^\mathrm{H} (\textbf{k}) = (U^\dagger \cdot \mathcal{S}^\mathrm{W} \cdot U)_{mn}`, where :math:`U_{mn}` is a unitary matrix which diagonalizes the interpolated Hamiltonian :math:`\overline{\mathcal{H}}_{mn}^\mathrm{W} (\textbf{k}) = \sum_\textbf{R} \mathcal{H}_{mn} (\textbf{R}) \cdot  e^{i \textbf{k} \cdot \textbf{R}}`.

This procedure is implemented in ``spinWannier``.

See the `Supplementary <https://pubs.acs.org/doi/suppl/10.1021/acs.nanolett.4c03029/suppl_file/nl4c03029_si_001.pdf>`_ of `L. Vojáček*, J. M. Dueñas* et al., Nano Letters (2024) <https://pubs.acs.org/doi/10.1021/acs.nanolett.4c03029>`_.