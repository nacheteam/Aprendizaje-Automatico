# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


################################################################################
##                             Ejercicio 1                                    ##
################################################################################

#------------------------------------------------------------------------------#
##                              Apartado 1                                    ##
#------------------------------------------------------------------------------#

def simula_unif(N, dim, rango):
    '''
    @brief Función que genera una nube de puntos de dimensión dada según una
    distribución uniforme
    @param N número de datos a generar
    @param dim Dimensión de los datos que queremos generar
    @parma rango Intervalo en el que se van a generar los datos
    @return Devuelve un numpy array con los datos
    '''
    return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gauss(N, dim, sigma):
    '''
    @brief Función que genera datos aleatorios de dimensión dada según una
    distribución normal o Gaussiana.
    @param N número de datos a generar
    @param dim Dimensión de los datos que queremos generar
    @param sigma Sigma de la distribución gaussiana en ambos ejes
    @return Devuelve un numpy array con los datos
    '''
    media = 0
    out = np.zeros((N,dim),np.float64)
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para
        # la primera columna se usará una N(0,sqrt(5)) y para la segunda N(0,sqrt(7))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    return out

def simula_recta(intervalo):
    '''
    @brief Simula una recta en un intervalo dado
    @param intervalo Intervalo en el  que va a estar la recta (intervalo x intervalo)
    @return Devuelve los parámetros a y b de la recta y = ax+b
    '''
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1)
    b = y1-a*x1

    return a,b

def ej1(N=50):
    # Simulamos datos según una distribución uniforme
    nube_unif = simula_unif(N,2,[-50,50])
    # Simulamos datos según una distribución Gaussiana
    nube_normal = simula_gauss(N,2,[5,7])

    # Pintamos la nube de puntos según una uniforme
    plt.scatter(nube_unif[:,0], nube_unif[:,1])
    plt.title("Nube con la uniforme")
    plt.show()

    # Pintamos la nube de puntos según una normal
    plt.scatter(nube_normal[:,0], nube_normal[:,1])
    plt.title("Nube con la normal")
    plt.show()

ej1()
