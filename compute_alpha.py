## imports

from cmath import *
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize



## utils

j=complex(0,1)

# parameters
c0=340
L=10
A=1
B=1
xi0=1/(c0**2)
eta0=1

# source
g = lambda y,omega : 0.1*exp(-(y**2)/8)*cos(omega*1)

# matérial (mélanine)
phi=0.99
sigma=14000
alpha_h=1.02
gamma_p=7/5

def param_material(phi, sigma, alpha_h, gamma_p, c0) :
    xi1=phi*gamma_p/(c0**2)
    eta1=phi/alpha_h
    a=(sigma*(phi**2)*gamma_p)/((c0**2)*alpha_h*1)
    return xi1, eta1, a

xi1, eta1, a=param_material(phi, sigma, alpha_h, gamma_p, c0)

# frequency
omega=10000



## error function

'''
The computation of the error function follows the same construction as the one presented in the statement of the exercice.
'''

def Gk(g,k) :
    dans_lintegrale = lambda y : (1/L)*g(y,omega)*exp(-j*2*k*y)
    g_k = integrate.quad(lambda x: np.real(dans_lintegrale(x)), 0, L)[0] + 1j * integrate.quad(lambda x: np.imag(dans_lintegrale(x)), 0, L)[0]
    return g_k

def Lambda0 (k, omega, xi0, eta0) :
    discriminant=k**2-xi0/eta0*omega**2
    if discriminant>0 :
        return sqrt(discriminant)
    return j*sqrt(-discriminant)

def Lambda1 (k, omega, xi1, eta1, a) :
    discriminant=k**2-xi1/eta1*omega**2
    R=sqrt(discriminant**2+(a*omega/eta1)**2)
    return complex(sqrt(discriminant+R), sqrt(-discriminant+R))/sqrt(2)

def F(lambda0, eta0, L) :
    def inner(x) :
        return (lambda0*eta0-x)*exp(-lambda0*L)+(lambda0*eta0+x)*exp(lambda0*L)
    return inner

def Chi(alpha, gk, lambda0, eta0, lambda1, eta1, L) :
    f=F(lambda0, eta0, L)
    p0=lambda0*eta0
    p1=lambda1*eta1
    return gk*((p0-p1)/f(p1)-(p0-alpha)/f(alpha))

def Gamma(alpha, gk, lambda0, eta0, lambda1, eta1, L) :
    f=F(lambda0, eta0, L)
    p0=lambda0*eta0
    p1=lambda1*eta1
    return gk*((p0+p1)/f(p1)-(p0+alpha)/f(alpha))

def Ek(g, k, omega, xi0, eta0, xi1, eta1, L, a, A, B, N) :
    lambda0=Lambda0(k, omega, xi0, eta0)
    lambda1=Lambda1(k, omega, xi1, eta1, a)
    gk=Gk(g,k)
    def inner(alpha) :
        chi=Chi(alpha, gk, lambda0, eta0, lambda1, eta1, L)
        gamma=Gamma(alpha, gk, lambda0, eta0, lambda1, eta1, L)
        if k**2-xi0/eta0*omega**2>0 :
            S=abs(chi)**2*(1-exp(-2*lambda0*L))+abs(gamma)**2*(exp(2*lambda0*L)-1)
            resk=(A+B*k**2)*(S/2/lambda0+2*L*(chi*gamma.conjugate()).real)+B*lambda0/2*S-2*B*lambda0**2*L*(chi*gamma.conjugate()).real
        else :
            T=(chi*gamma.conjugate()*(1-exp(-2*lambda0*L))).imag
            resk=(A+B*k**2)*(L*(abs(chi)**2+abs(gamma)**2)+j/lambda0*T)+B*L*lambda0**2*(abs(chi)**2+abs(gamma)**2)+j*B*lambda0*T
        if resk.imag!=0 :
            raise 'ek pas reel'
        return resk.real
    return inner

def E(g, omega, xi0, eta0, xi1, eta1, L, a, A, B, N) :
    '''
    the infinite sum is approximated by a sum from n=-N to n=N
    '''
    def inner(alpha) :
        res=Ek(g, 0, omega, xi0, eta0, xi1, eta1, L, a, A, B, N)(alpha)
        for n in range(1, N+1) :
            ek=Ek(g, n*pi/L, omega, xi0, eta0, xi1, eta1, L, a, A, B, N)
            e_k=Ek(g, -n*pi/L, omega, xi0, eta0, xi1, eta1, L, a, A, B, N)
            res+=ek(alpha)+e_k(alpha)
        return abs(res)
    return inner

# print(E(g, omega, xi0, eta0, xi1, eta1, L, a, A, B, 100)(complex(0,0)))



## mminimization of the error

def E_R(g, omega, xi0, eta0, xi1, eta1, L, a, A, B, N) :
    '''
    scypy.optimize minimize function can only minize function of R^n. We therefore need switch from a function of C to a function of R^2.
    '''
    e=E(g, omega, xi0, eta0, xi1, eta1, L, a, A, B, N)
    def inner(var) :
        return e(complex(var[0], var[1]))
    return inner

def solve_alpha(g, omega, xi0, eta0, xi1, eta1, L, a, A, B, N, x_init) :
    e_R=E_R(g, omega, xi0, eta0, xi1, eta1, L, a, A, B, N)
    result = minimize(e_R, x_init)
    return result['x']

# print(solve_alpha(g, omega, xi0, eta0, xi1, eta1, L, a, A, B, 50, [0,0]))



## plots

def graph(g, Omega_list, xi0, eta0, xi1, eta1, L, a, A, B, N, x_init) :
    Alpha_re = []
    Alpha_im = []
    for omega in Omega_list:
        x,y=solve_alpha(g, omega, xi0, eta0, xi1, eta1, L, a, A, B, N, x_init)
        Alpha_re.append(x)
        Alpha_im.append(y)
    plt.plot(Omega_list, Alpha_re)
    plt.show()
    plt.plot(Omega_list, Alpha_im)
    plt.show()


nb_points = 100
Xs, Ys = np.meshgrid(np.linspace(-100, 100, nb_points), np.linspace(-100, 100, nb_points))
Xs = Xs.flatten()
Ys = Ys.flatten()
vars2 = np.zeros_like(Xs)

'''
#calculation of the variance
for i in range(np.size(Xs)):
    X,Y = Xs[i], Ys[i]
    vars2[i] = E_R(g, omega, xi0, eta0, xi1, eta1, L, a, A, B, 50)([X, Y])
#plot of the variance of r as a fonction of X and Y.
plt.scatter(Xs,Ys,c=vars2,cmap='gist_rainbow', vmin=0, vmax=10)
plt.xlabel(r"$Re(alpha)}$")
plt.ylabel(r"$Im(alpha)$")
plt.title(r"Error as a fonction of the real and imaginary part of alpha")
plt.colorbar()
plt.show()
'''


## different sources

# matérial (mélanine)
phi=0.99
sigma=14000
alpha_h=1.02
gamma_p=7/5

xi1, eta1, a=param_material(phi, sigma, alpha_h, gamma_p, c0)

#Problem 1: Metro noise

g = lambda y,omega : 0.1*exp(-(y**2)/8)*cos(omega*1)

Omega_list = [10000, 20000, 30000, 40000]
N=50
x_init=[0,0]

# graph(g, Omega_list, xi0, eta0, xi1, eta1, L, a, A, B, N, x_init)

#Problem 2: Industrial generator noise

g = lambda y,omega : 0.5*sin(omega*1)

Omega_list = [10000, 20000, 30000, 40000]
N=50
x_init=[0,0]

# graph(g, Omega_list, xi0, eta0, xi1, eta1, L, a, A, B, N, x_init)

#Problème  : Helicopter noise

g = lambda y,omega : 1*exp(-(y**2)/1)*sin(omega*1) + 0.7*exp(-((y-3)**2)/0.25)*sin(omega*1.5)

Omega_list = [10000, 20000, 30000, 40000]
N=50
x_init=[0,0]

# graph(g, Omega_list, xi0, eta0, xi1, eta1, L, a, A, B, N, x_init)
    


## different materials

# source
g = lambda y,omega : 0.1*exp(-(y**2)/8)*cos(omega*1)

#Problem 1 : Melamine foam

phi=0.99
sigma=14000
alpha_h=1.02
gamma_p=7/5

xi1, eta1, a=param_material(phi, sigma, alpha_h, gamma_p, c0)

Omega_list = [10000, 20000, 30000, 40000]
N=50
x_init=[0,0]

# graph(g, Omega_list, xi0, eta0, xi1, eta1, L, a, A, B, N, x_init)


#Problem 2 : ISOREL

phi=0.70
sigma=142300
alpha_h=1.15
gamma_p=7/5

xi1, eta1, a=param_material(phi, sigma, alpha_h, gamma_p, c0)

Omega_list = [10000, 20000, 30000, 40000]
N=50
x_init=[0,0]

# graph(g, Omega_list, xi0, eta0, xi1, eta1, L, a, A, B, N, x_init)

#Problem 3 : B5

phi=0.10
sigma=2124000
alpha_h=1.22
gamma_p=7/5

xi1, eta1, a=param_material(phi, sigma, alpha_h, gamma_p, c0)

Omega_list = [10000, 20000, 30000, 40000]
N=50
x_init=[0,0]

# graph(g, Omega_list, xi0, eta0, xi1, eta1, L, a, A, B, N, x_init)


def compute_alpha(L, omega, g) :
    N_iter=20
    x_init=[0,0]
    # parameters
    c0=340
    A=1
    B=1
    xi0=1/(c0**2)
    eta0=1
    # matérial (mélanine)
    phi=0.99
    sigma=14000
    alpha_h=1.02
    gamma_p=7/5
    xi1, eta1, a=param_material(phi, sigma, alpha_h, gamma_p, c0)
    return solve_alpha(g, omega, xi0, eta0, xi1, eta1, L, a, A, B, N_iter, x_init)