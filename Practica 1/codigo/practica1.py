# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

################################################################################
##                             Ejercicio 1                                    ##
################################################################################

#------------------------------------------------------------------------------#
##                              Apartado 1                                    ##
#------------------------------------------------------------------------------#

# Función de la que queremos hallar el mínimo.
def E(u,v):
    return (u**2*np.exp(v)-2*v**2*np.exp(-u))**2

# Derivada parcial de E con respecto a u
def dEu(u,v):
    return 2*np.sqrt(E(u,v))*(2*u*np.exp(v)+2*v**2*np.exp(-u))

# Derivada parcial de E con respecto a v
def dEv(u,v):
    return 2*np.sqrt(E(u,v))*(u**2*np.exp(v)-4*v*np.exp(-u))

# Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

# Implementación del método gradiente descendente
def gradient_descent(w_init,learning_rate,max_iter,tol):
    w_before=w_init
    w = w_before-learning_rate*gradE(w_before[0],w_before[1])
    iterations=1
    while iterations<max_iter and np.linalg.norm(E(w[0],w[1])-E(w_before[0],w_before[1]))>=tol:
        w_before=w
        w = w_before-learning_rate*gradE(w_before[0],w_before[1])
        iterations+=1
    return w, iterations

#------------------------------------------------------------------------------#
##                               Apartado 2                                   ##
#------------------------------------------------------------------------------#

def apartado2():
    eta = 0.01
    maxIter = 10000000000
    error2get = 1e-14
    initial_point = np.array([1.0,1.0])
    w, it = gradient_descent(initial_point,eta,maxIter,error2get)
    print("El número de iteraciones empleado ha sido de: " + str(it))
    print("El mínimo encontrado por gradiente desdendente ha sido: ")
    print("x: (" + str(w[0]) + ")")
    print("y: (" + str(w[1]) + ")")

apartado2()
