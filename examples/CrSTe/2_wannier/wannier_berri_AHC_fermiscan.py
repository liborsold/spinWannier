import wannierberri as wberri
import numpy as np
import os

# switch off parallelization of numpy; see https://wannier-berri.org/install.html
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


# ======= USER INPUT ===========

num_proc = 32
fermi_energies = np.linspace(E_min,E_max,41)
seedname = 'wannier90'
adaptive_iters = 30   # number of adaptive iterations
length = 1000

# ==============================

def main():


    system=wberri.System_w90(seedname, berry=True)

    grid=wberri.Grid(system,length=length)

    parallel=wberri.Parallel(num_cpus=num_proc)

    wberri.integrate(system, grid,
                Efermi=fermi_energies,
                smearEf=10, # 10K
                quantities=["ahc","dos","cumdos"],
                parallel = parallel,
                adpt_num_iter=adaptive_iters,  # number of adaptive refinement steps
                fout_name='AHC_fermiscan')


if __name__ == "__main__":
    main()
