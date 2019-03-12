# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from tabulate import tabulate

np.random.seed(1)

################################################################################
##                             Ejercicio 1                                    ##
################################################################################

#------------------------------------------------------------------------------#
##                              Apartado 1                                    ##
#------------------------------------------------------------------------------#
def evaluate(E,symbols,w):
    point = dict(zip(symbols,w))
    return N(E.subs(point))

def gradiente(E,symbols,w):
    gradiente = []
    for i in range(len(symbols)):
        dife = diff(E,symbols[i])
        gradiente.append(evaluate(dife,symbols,w))

    return np.array(gradiente)

# Implementación del método gradiente descendente
def gradient_descent(w_init,learning_rate,max_iter,tol,E,symbols,ret_values=False):
    w_before=w_init
    w = w_before-learning_rate*gradiente(E,symbols,w_before)
    # Cuidado que no hay el mismo número de elementos que iteraciones, para la primera iteración
    # introducimos dos valores, no uno
    func_values = [evaluate(E,symbols,w_init),evaluate(E,symbols,w)]
    iterations=1
    while iterations<max_iter and np.absolute(evaluate(E,symbols,w)-evaluate(E,symbols,w_before))>=tol:
        w_before=w
        w = w_before-learning_rate*gradiente(E,symbols,w_before)
        func_values.append(evaluate(E,symbols,w))
        iterations+=1
    if ret_values:
        return w, iterations, func_values
    else:
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
    print("#################################\nEjercicio 1, apartado 2\n#################################\n\n")
    print("El número de iteraciones empleado ha sido de: " + str(it))
    print("El mínimo encontrado por gradiente desdendente ha sido: ")
    print("x: " + str(w[0]))
    print("y: " + str(w[1]))
    input("Presione ENTER para continuar")

#------------------------------------------------------------------------------#
##                               Apartado 3                                   ##
#------------------------------------------------------------------------------#

def apartado3a(eta):
    maxIter = 50
    error2get = 1e-14
    initial_point = np.array([0.1,0.1])
    x,y = symbols('x y',real=True)
    symbol = [x,y]
    expr = x**2 + 2*y**2 + 2*sin(2*pi*x)*sin(2*pi*y)
    w, it, values = gradient_descent(initial_point,eta,maxIter,error2get,expr,symbol,ret_values=True)
    print("#################################\nEjercicio 1, apartado 3 a\n#################################\n\n")
    print("Estamos utilizando eta = " + str(eta))
    print("El número de iteraciones empleado ha sido de: " + str(it))
    print("El mínimo encontrado por gradiente desdendente ha sido: ")
    print("x: " + str(w[0]))
    print("y: " + str(w[1]))

    plt.plot(list(range(len(values))),values,label="Valores de la función para cada iteración")
    plt.legend()
    plt.show()
    input("Presione ENTER para continuar")

def apartado3b():
    eta=0.01
    maxIter = 50
    error2get = 1e-14
    x,y = symbols('x y',real=True)
    symbol = [x,y]
    expr = x**2 + 2*y**2 + 2*sin(2*pi*x)*sin(2*pi*y)
    ws_init = [np.array([0.1,0.1]),np.array([1,1]),np.array([-0.5,-0.5]),np.array([-1,-1])]
    results = []
    for w_init in ws_init:
        w, it = gradient_descent(w_init,eta,maxIter,error2get,expr,symbol)
        results.append([w_init,w,evaluate(expr,symbol,w)])

    print("#################################\nEjercicio 1, apartado 3 b\n#################################\n\n")
    print("Tabla con los valores de los mínimos encontrados: \n")
    print(tabulate(results,headers=["Punto inicial", "Mínimo encontrado", "Valor en el mínimo"]))


apartado2()
apartado3a(0.01)
apartado3a(0.1)
apartado3b()
