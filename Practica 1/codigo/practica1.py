# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sympy import *

np.random.seed(1)

################################################################################
##                             Ejercicio 1                                    ##
################################################################################

#------------------------------------------------------------------------------#
##                              Apartado 1                                    ##
#------------------------------------------------------------------------------#
def evaluate(E,symbols,w):
    value = E.subs(symbols[0],w[0])
    for j in range(1,len(symbols)):
        new_value = value.subs(symbols[j],w[j])
        value = new_value
    return N(value)

def gradiente(E,symbols,w):
    gradiente = []
    for i in range(len(symbols)):
        dife = diff(E,symbols[i])
        gradiente.append(evaluate(dife,symbols,w))

    return np.array(gradiente)

# Implementación del método gradiente descendente
def gradient_descent(w_init,learning_rate,max_iter,tol,E,symbols):
    w_before=w_init
    w = w_before-learning_rate*gradiente(E,symbols,w_before)
    iterations=1
    while iterations<max_iter and np.absolute(evaluate(E,symbols,w)-evaluate(E,symbols,w_before))>=tol:
        w_before=w
        w = w_before-learning_rate*gradiente(E,symbols,w_before)
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
    u,v = symbols('u v',real=True)
    symbol = [u,v]
    expr = (u**2*exp(v)-2*v**2*exp(-u))**2
    w, it = gradient_descent(initial_point,eta,maxIter,error2get,expr,symbol)
    print("El número de iteraciones empleado ha sido de: " + str(it))
    print("El mínimo encontrado por gradiente desdendente ha sido: ")
    print("x: " + str(w[0]))
    print("y: " + str(w[1]))

#------------------------------------------------------------------------------#
##                               Apartado 3                                   ##
#------------------------------------------------------------------------------#

def apartado3():
    print("hola")

apartado2()
