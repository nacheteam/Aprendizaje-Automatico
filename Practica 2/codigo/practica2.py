# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123456789)

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

def ej1ap1(N=50):
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

#ej1ap1()

#------------------------------------------------------------------------------#
##                              Apartado 2                                    ##
#------------------------------------------------------------------------------#

def fAp2(x,y,a,b):
    return np.sign(y-a*x-b)

def ej1ap2(N=50):
    # Generamos una nube de puntos según una uniforme
    nube_unif = simula_unif(N,2,[-50,50])
    # Simulamos una recta
    a,b = simula_recta([-50,50])
    # Calculamos las etiquetas usando la función f(x,y) = y-a*x -b
    labels = np.array([])
    for punto in nube_unif:
        labels = np.append(labels,fAp2(punto[0],punto[1],a,b))

    # Separamos los datos según las etiquetas
    datosA = np.array([nube_unif[i] for i in range(len(nube_unif)) if labels[i]==-1])
    datosB = np.array([nube_unif[i] for i in range(len(nube_unif)) if labels[i]==1])

    # Dibujamos los datos por colores según etiquetas
    plt.scatter(datosA[:,0], datosA[:,1], c="green", label="Datos con etiqueta -1")
    plt.scatter(datosB[:,0], datosB[:,1], c="red", label="Datos con etiqueta 1")
    plt.plot(list(range(-51,51)),list(map(lambda x,a=a,b=b:a*x+b, list(range(-51,51)))),c="blue", label="Recta divisora")
    plt.title("Nube uniforme separada")
    plt.legend()
    plt.show()

    # Cogemos los índices de los datos con etiquetas positivas y negativas
    labelsPos = np.array([i for i in range(len(labels)) if labels[i]==1])
    labelsNeg = np.array([i for i in range(len(labels)) if labels[i]==-1])

    # Calculamos un conjunto de subindices de etiquetas positivas y negativas
    ind1 = np.random.choice(len(labelsPos),int(0.1*len(labelsPos)), replace=False)
    ind2 = np.random.choice(len(labelsNeg),int(0.1*len(labelsNeg)), replace=False)
    # Introducimos ruido
    labels[labelsPos[ind1]] = -labels[labelsPos[ind1]]
    labels[labelsNeg[ind2]] = -labels[labelsNeg[ind2]]

    # Separamos los datos de nuevo por etiquetas
    datosA = np.array([nube_unif[i] for i in range(len(nube_unif)) if labels[i]==-1])
    datosB = np.array([nube_unif[i] for i in range(len(nube_unif)) if labels[i]==1])

    # Pintamos los datos con ruido
    plt.scatter(datosA[:,0], datosA[:,1], c="green", label="Datos con etiqueta -1")
    plt.scatter(datosB[:,0], datosB[:,1], c="red", label="Datos con etiqueta 1")
    plt.plot(list(range(-51,51)),list(map(lambda x,a=a,b=b:a*x+b, list(range(-51,51)))),c="blue", label="Recta divisora")
    plt.title("Nube uniforme con ruido")
    plt.legend()
    plt.show()

ej1ap2()
