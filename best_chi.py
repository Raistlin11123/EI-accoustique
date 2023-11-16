# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot as plt
import numpy as np
import os

# MRG packages
import _env
import preprocessing
import processing
import postprocessing
#import solutions
np.set_printoptions(threshold=np.inf)

from demo_control_polycopie2023 import your_optimization_procedure_multi, chi_zero_ou_un

if __name__ == '__main__':

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry

    N = 40  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 1 # level of the fractal
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    #kx=ky= (sqrt(2)pi/340)*f
    freq= np.linspace(1500,3500,10)
    kx = -((np.sqrt(2)*np.pi)/340)*freq
    ky = kx
    #kx = 5*(-1.0)
    #ky = 5*(-1.0)
    wavenumber = np.sqrt(kx**2 + ky**2)  # wavenumber
    omega= 340*wavenumber
    min_energies = np.zeros(freq.shape[0])

    

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------

    # -- define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)
    

    # -- define absorbing material
    import compute_alpha
    #Alpha = compute_alpha.compute_alpha(N*spacestep, omega, g)
    Alpha = (1-1j)
    alpha_rob = Alpha * chi
    

    # -- set parameters for optimization
    S = 0  # surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1

    V_0 = 1  # initial volume of the domain
    V_obj = np.sum(np.sum(chi)) / S  # constraint on the density
    mu = 10 ** -3  # initial gradient step
    mu1 = 10**(-5)  # parameter of the volume functional

    f_dirs = []
    for i in range(wavenumber.shape[0]):
        #g = lambda y,omega : 0.1*np.exp(-(y**2)/8)*np.cos(omega*1)
        g = lambda x, omega : np.exp(complex(0,1)*(kx[i]*x+ky[i]*0))

        # -- define boundary conditions
        # planar wave defined on top
        f_dir[:, :] = 0.0
        for j in range(N) :
            f_dir[0, j] = g(spacestep*(j-N/2), omega[i])
        f_dirs.append(f_dir)
        #f_dir[0, 0:N] = g(0, omega)

    # spherical wave defined on top
    #f_dir[:, :] = 0.0
    #f_dir[0, int(N/2)] = 10.0



    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- compute finite difference solution
    u0s = []
    for i in range(wavenumber.shape[0]):
        u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber[i], f, f_dirs[i], f_neu, f_rob,
                            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        chi0 = chi.copy()
        u0 = u.copy()
        u0s.append(u0)
    

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- compute optimization

    # energy = np.zeros((100+1, 1), dtype=np.float64)
    chi, energy, u, grad = your_optimization_procedure_multi(domain_omega, spacestep, wavenumber, f, f_dirs, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, mu, chi, V_obj)

    # --- en of optimization
    chin = chi.copy()
    sum_un = u.copy()

    min_energies[i] = energy[-1]



    # -- plot chi, u, and energy
    postprocessing._plot_uncontroled_solution(sum(u0s), chi0)
    postprocessing._plot_controled_solution(sum_un, chi_zero_ou_un(chin,domain_omega,V_obj))
    err = sum_un - u0
    postprocessing._plot_error(err)
    postprocessing._plot_energy_history(energy)

    print('End.')