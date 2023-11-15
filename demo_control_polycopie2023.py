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

def compute_J_prim(alpha, u, p):
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
                #print('chi1:',chi[i + 1, j])
            if b == 2:
                #print(i - 1, j, "-----", "i - 1, j")
                chi[i - 1, j] = chi[i - 1, j] - mu * grad[i, j]
                #print('chi2:',chi[i - 1, j])
            if c == 2:
                #print(i, j + 1, "-----", "i , j + 1")
                chi[i, j + 1] = chi[i, j + 1] - mu * grad[i, j]
                #print('chi3:',chi[i, j+1])
            if d == 2:
                #print(i, j - 1, "-----", "i , j - 1")
                chi[i, j - 1] = chi[i, j - 1] - mu * grad[i,j]
                #print('chi4:',chi[i, j-1])
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
    list_rob=[]
    S = 0
    for i in range(M):
        for j in range(N):
            if domain[i, j] == _env.NODE_ROBIN:
                S = S + 1
                list_rob.append([(i,j),0])

    B = chi.copy()
    l = 0
    chi = processing.set2zero(chi, domain)
    V = np.sum(np.sum(chi)) / S
    debut = -np.max(chi)
    fin = np.max(chi)
    ecart = fin - debut
    
    # We use dichotomy to find a constant such that chi^{n+1}=max(0,min(chi^{n}+l,1)) is an element of the admissible space
    while ecart > 10 ** -4:
        #print("l:",l)
        # calcul du milieu 
        l = (debut + fin) / 2
        for i in range(M):
            for j in range(N):
                chi[i, j] = np.maximum(0, np.minimum(B[i, j] + l, 1))
        chi = processing.set2zero(chi, domain)
        V = np.sum(np.sum(chi)) / S
        if V > V_obj:
            fin = l
        else:
            debut = l
        ecart = fin - debut
        #print("écart", ecart)
        #print('le volume est', V, 'le volume objectif est', V_obj)

#    chi=chi_zero_ou_un (M, N, chi, S, V_obj, list_rob)

    return chi

#def chi_zero_ou_un (M, N, chi, S, V_obj, list_rob) :
#    for el in list_rob :
#        el[1]=chi[el[0]]
#    list_rob=sorted(list_rob, key=lambda couple : couple[1], reverse=True)
#    chiprim = np.zeros((M,N))
#    for i in range(int(S*V_obj)) :
#        chiprim[list_rob[i][0]]=1
#    return chiprim



def your_optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, mu, chi, V_obj):
    """This function return the optimized density.

    Parameter:
        cf solvehelmholtz's remarks
        Alpha: complex, it corresponds to the absorbtion coefficient;
        mu: float, it is the initial step of the gradient's descent;
        V_obj: float, it characterizes the volume constraint on the density chi.
    """
    k = 0
    (M, N) = np.shape(domain_omega)
    numb_iter = 100
    energy = np.zeros((numb_iter, 1), dtype=np.float64)
    is_good = True
    while k < numb_iter:
        print("mu entrée de boucle:", mu)
        if not is_good:
            mu *= 2
        print(f"k={k}")
        #print('1. computing solution of Helmholtz problem, i.e., u')
        u=processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        #print('2. computing solution of adjoint problem, i.e., p')
        p=processing.solve_helmholtz(domain_omega, spacestep, wavenumber, np.conjugate(-2*u), np.zeros((M,N)), f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        #print('3. computing objective function, i.e., energy')
        J=your_compute_objective_function(u,spacestep)
        energy[k]=J
        print("energie = ",J)
        Jprim=compute_J_prim(Alpha, u, p)
        #print('4. computing parametric gradient')
        ene=J
        grad = -Jprim
        #print("Jprim=",Jprim)
        #print("grad=",grad)
        is_good = True
        while ene >= energy[k] and mu > 10 ** -5:
            #print('    a. computing gradient descent')
            chi = compute_gradient_descent(chi, grad, domain_omega, mu) #chi_k+1 sans projection (l=0)

            #print('    b. computing projected gradient')
            chi = compute_projected(chi, domain_omega, V_obj)
            alpha_rob = Alpha*chi
            #print('    c. computing solution of Helmholtz problem, i.e., u')
            u=processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
            

            #print('    d. computing objective function, i.e., energy (E)')
            ene = your_compute_objective_function(u,spacestep)
            bool_a=ene<J
            if bool_a:
                # The step is increased if the energy decreased
                mu = mu * 1.1
                print("mu bonne éné:", mu)
                print("éné:",ene)
                is_good = True
            else:
                # The step is decreased is the energy increased
                mu = mu / 2
                print("mu mauvaise éné:", mu)
                print("éné:",ene)
                is_good = False
                
        k += 1

    print('end. computing solution of Helmholtz problem, i.e., u')

    return chi, energy, u, grad


def your_compute_objective_function(u,spacestep):
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
    #((re² + im² )step²)^1/2
    energy = np.linalg.norm(u)*spacestep

    return energy



if __name__ == '__main__':

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry

    N = 50  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 0 # level of the fractal
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    #kx=ky= (sqrt(2)pi/340)*f
    f= np.linspace(1500,3500,10)
    kx = -((np.sqrt(2)*np.pi)/340)*f
    ky = kx
    #kx = 5*(-1.0)
    #ky = 5*(-1.0)
    wavenumber = np.sqrt(kx**2 + ky**2)  # wavenumber
    omega= 340*wavenumber
    
    #g = lambda y,omega : 0.1*np.exp(-(y**2)/8)*np.cos(omega*1)
    g = lambda x, omega : np.exp(complex(0,1)*(kx*x+ky*0))

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
    # -- define boundary conditions
    # planar wave defined on top
    f_dir[:, :] = 0.0
    for j in range(N) :
        f_dir[0, j] = g(spacestep*(j-N/2), omega)
    #f_dir[0, 0:N] = g(0, omega)

    # spherical wave defined on top
    #f_dir[:, :] = 0.0
    #f_dir[0, int(N/2)] = 10.0

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
    mu = 10 ** -1  # initial gradient step
    mu1 = 10**(-5)  # parameter of the volume functional

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- compute finite difference solution
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    chi0 = chi.copy()
    u0 = u.copy()

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


    # -- plot chi, u, and energy
    postprocessing._plot_uncontroled_solution(u0, chi0)
    postprocessing._plot_controled_solution(un, chin)
    err = un - u0
    postprocessing._plot_error(err)
    postprocessing._plot_energy_history(energy)

    print('End.')
