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

def compute_J_prim(alpha, u, p) :
    #alpha complexe
    M,N=np.shape(u)
    res=np.zeros((M,N))

    for i in range(M) :
        for j in range(N) :
            res[i,j]=np.real(-alpha*u[i,j]*p[i,j])
    return res


def BelongsInteriorDomain(node):
                if (node < 0):
                    return 1
                if node == 3:
                    #print("Robin")
                    return 2
                else:
                    return 0


def compute_gradient_descent(chi, grad, domain, mu):
    """This function makes the gradient descent.
    This function has to be used before the 'Projected' function that will project
    the new element onto the admissible space.
    :param chi: density of absorption define everywhere in the domain
    :param grad: parametric gradient associated to the problem
    :param domain: domain of definition of the equations
    :param mu: step of the descent
    :type chi: np.array((M,N), dtype=float64
    :type grad: np.array((M,N), dtype=float64)
    :type domain: np.array((M,N), dtype=int64)
    :type mu: float
    :return chi:
    :rtype chi: np.array((M,N), dtype=float64

    .. warnings also: It is important that the conditions be expressed with an "if",
            not with an "elif", as some points are neighbours to multiple points
            of the Robin frontier.
    """

    (M, N) = np.shape(domain)
    # for i in range(0, M):
    # 	for j in range(0, N):
    # 		if domain_omega[i, j] != _env.NODE_ROBIN:
    # 			chi[i, j] = chi[i, j] - mu * grad[i, j]
    # # for i in range(0, M):
    # 	for j in range(0, N):
    # 		if preprocessing.is_on_boundary(domain[i , j]) == 'BOUNDARY':
    # 			chi[i,j] = chi[i,j] - mu*grad[i,j]
    # print(domain,'jesuisla')
    #chi[50,:] = chi[50,:] - mu*grad[50,:]
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            #print(i,j)
            #chi[i,j] = chi[i,j] - mu * grad[i,j]
            a = BelongsInteriorDomain(domain[i + 1, j])
            b = BelongsInteriorDomain(domain[i - 1, j])
            c = BelongsInteriorDomain(domain[i, j + 1])
            d = BelongsInteriorDomain(domain[i, j - 1])
            if a == 2:
                #print(i+1,j, "-----", "i+1,j")

                chi[i + 1, j] = chi[i + 1, j] - mu * grad[i, j]
            if b == 2:
                #print(i - 1, j, "-----", "i - 1, j")
                chi[i - 1, j] = chi[i - 1, j] - mu * grad[i, j]
            if c == 2:
                #print(i, j + 1, "-----", "i , j + 1")
                chi[i, j + 1] = chi[i, j + 1] - mu * grad[i, j]
            if d == 2:
                #print(i, j - 1, "-----", "i , j - 1")
                chi[i, j - 1] = chi[i, j - 1] - mu * grad[i,j]
    return chi


def compute_projected(chi, domain, V_obj):
    """This function performs the projection of $\chi^n - mu*grad

    To perform the optimization, we use a projected gradient algorithm. This
    function caracterizes the projection of chi onto the admissible space
    (the space of $L^{infty}$ function which volume is equal to $V_{obj}$ and whose
    values are located between 0 and 1).

    :param chi: density matrix
    :param domain: domain of definition of the equations
    :param V_obj: characterizes the volume constraint
    :type chi: np.array((M,N), dtype=float64)
    :type domain: np.array((M,N), dtype=complex128)
    :type float: float
    :return:
    :rtype:
    """

    (M, N) = np.shape(domain)
    S = 0
    for i in range(M):
        for j in range(N):
            if domain[i, j] == _env.NODE_ROBIN:
                S = S + 1

    B = chi.copy()
    l = 0
    chi = processing.set2zero(chi, domain)
    V = np.sum(np.sum(chi)) / S
    debut = -np.max(np.abs(chi))
    fin = np.max(np.abs(chi))
    ecart = fin - debut
    
    # We use dichotomy to find a constant such that chi^{n+1}=max(0,min(chi^{n}+l,1)) is an element of the admissible space
    while ecart > 10 ** -4:
        # calcul du milieu 
        l = (debut + fin) / 2
        for i in range(M):
            for j in range(N):
                chi[i, j] = np.maximum(0, np.minimum(B[i, j] + l, 1))
        chi = processing.set2zero(chi, domain)
        V = sum(sum(chi)) / S
        if V > V_obj:
            fin = l
        else:
            debut = l
        ecart = fin - debut
        #print("écart", ecart)
        #print('le volume est', V, 'le volume objectif est', V_obj)

    return chi



def your_optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, mu, chi, V_obj):
    #omega/wavelenght attention
    """This function return the optimized density.

    Parameter:
        cf solvehelmholtz's remarks
        Alpha: complex, it corresponds to the absorbtion coefficient;
        mu: float, it is the initial step of the gradient's descent;
        V_obj: float, it characterizes the volume constraint on the density chi.
    """
    k = 0
    (M, N) = np.shape(domain_omega)
    numb_iter = 5 
    energy = np.zeros((numb_iter, 1), dtype=np.float64)
    while k < numb_iter and mu > 10**(-5):
        print('1. computing solution of Helmholtz problem, i.e., u')
        u=processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        print('2. computing solution of adjoint problem, i.e., p')
        p=processing.solve_helmholtz(domain_omega, spacestep, omega, np.conjugate(-2*u), np.zeros_like(domain_omega), f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        print('3. computing objective function, i.e., energy')
        J=your_compute_objective_function(u)
        energy[k]=J
        Jprim=compute_J_prim(Alpha, u, p)
        print('4. computing parametric gradient')
        ene=J
        
        while ene >= energy[k] and mu > 10 ** -5:
            grad = Jprim

            print('    a. computing gradient descent')
            chi = compute_gradient_descent(chi, grad, domain_omega, mu) #chi_k+1 sans projection (l=0)

            print('    b. computing projected gradient')
            chi = compute_projected(chi, domain_omega, V_obj)
            
            print('    c. computing solution of Helmholtz problem, i.e., u')
            u=processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
            

            print('    d. computing objective function, i.e., energy (E)')

            ene = your_compute_objective_function(u)
            bool_a=ene<J
            if bool_a:
                # The step is increased if the energy decreased
                mu = mu * 1.1
            else:
                # The step is decreased is the energy increased
                mu = mu / 2

        alpha_rob = Alpha*chi
        k += 1

    print('end. computing solution of Helmholtz problem, i.e., u')
    grad=0
    
    return chi, energy, u, grad


def your_compute_objective_function(u):
    """
    This function compute the objective function:
    J(u,domain_omega)= \int_{domain_omega}||u||^2 

    Parameter:
        domain_omega: Matrix (NxP), it defines the domain and the shape of the
        Robin frontier;
        u: Matrix (NxP), it is the solution of the Helmholtz problem, we are
        computing its energy;
        spacestep: float, it corresponds to the step used to solve the Helmholtz
        equation.
    """

    (M,N) = np.shape(u)
    energy = 0
    for i in range(M) :
        for j in range(N) :
            energy+=abs(u[i,j])**2

    return energy


if __name__ == '__main__':

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry
    N = 100  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 1 # level of the fractal
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    wavenumber = np.sqrt(kx**2 + ky**2)  # wavenumber
    # wavenumber = 10.0

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)
    #print("domain_omega",domain_omega)

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- define boundary conditions
    # planar wave defined on top
    f_dir[:, :] = 0.0
    f_dir[0, 0:N] = 1.0
    '''
    faut que ce soit égal à g sur Gamma_dir
    '''
    # spherical wave defined on top
    #f_dir[:, :] = 0.0
    #f_dir[0, int(N/2)] = 10.0

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    # -- define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)
    

    # -- define absorbing material
    # Alpha = 10.0 - 10.0 * 1j
    # -- this is the function you have written during your project
    import compute_alpha
    Alpha = compute_alpha.solve_alpha(...)
    '''
    à modifier également
    '''
    #Alpha=1+1j
    alpha_rob = Alpha * chi

    # -- set parameters for optimization
    S = 0  # surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1

    V_0 = 1  # initial volume of the domain
    V_obj = np.sum(np.sum(chi)) / S  # constraint on the density
    mu = 5  # initial gradient step
    mu1 = 10**(-5)  # parameter of the volume functional

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- compute finite difference solution
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    chi0 = chi.copy()
    u0 = u.copy()

    print("shape chi0:",chi0.shape)
    print("shape u0:",np.real(u0[0:N]))
    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- compute optimization
    # energy = np.zeros((100+1, 1), dtype=np.float64)
    chi, energy, u, grad = your_optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, mu, chi, V_obj)
    # --- en of optimization
    chin = chi.copy()
    un = u.copy()

    print("shape chin:",chin.shape)
    print("shape ub:",un.shape)

    # -- plot chi, u, and energy
    postprocessing._plot_uncontroled_solution(u0, chi0)
    postprocessing._plot_controled_solution(un, chin)
    err = un - u0
    postprocessing._plot_error(err)
    postprocessing._plot_energy_history(energy)

    print('End.')
