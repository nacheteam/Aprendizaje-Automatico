# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123456789)

# VARIABLES GLOBALES
#--------------------------------------
nube_unif = None
labels_sin_ruido = None
labels_con_ruido = None

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
    # Declaramos las variables globales para su uso
    global nube_unif, labels_sin_ruido, labels_con_ruido
    # Generamos una nube de puntos según una uniforme
    nube_unif = simula_unif(N,2,[-50,50])
    # Simulamos una recta
    a,b = simula_recta([-50,50])
    # Calculamos las etiquetas usando la función f(x,y) = y-a*x -b
    labels = np.array([])
    for punto in nube_unif:
        labels = np.append(labels,fAp2(punto[0],punto[1],a,b))

    # Actualizamos la variable global
    labels_sin_ruido = np.copy(labels)

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

    # Actualizamos la variable global
    labels_con_ruido = np.copy(labels)

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

#------------------------------------------------------------------------------#
##                              Apartado 3                                    ##
#------------------------------------------------------------------------------#

def f1(x):
    return (x[0]-10)**2 + (x[1]-20)**2 - 400

def f2(x):
    return 0.5*(x[0]+10)**2 + (x[1]-20)**2 - 400

def f3(x):
    return 0.5*(x[0]-10)**2 - (x[1]+20)**2 - 400

def f4(x):
    return x[1] - 20*x[0]**2 - 5*x[0] + 3

def ej1ap3():
    datosA = np.array([nube_unif[i] for i in range(len(nube_unif)) if labels_con_ruido[i]==-1])
    datosB = np.array([nube_unif[i] for i in range(len(nube_unif)) if labels_con_ruido[i]==1])

    x = np.arange(-50,50,0.1)
    y = np.arange(-50,50,0.1)
    xx,yy = np.meshgrid(x,y)

    plt.scatter(datosA[:,0], datosA[:,1], c="green", label="Datos con etiqueta -1")
    plt.scatter(datosB[:,0], datosB[:,1], c="red", label="Datos con etiqueta 1")
    plt.contour(x,y,f1([xx,yy]),[0])
    plt.title("Nube uniforme con ruido usando la función 1")
    plt.legend()
    plt.show()

    plt.scatter(datosA[:,0], datosA[:,1], c="green", label="Datos con etiqueta -1")
    plt.scatter(datosB[:,0], datosB[:,1], c="red", label="Datos con etiqueta 1")
    plt.contour(x,y,f2([xx,yy]),[0])
    plt.title("Nube uniforme con ruido usando la función 2")
    plt.legend()
    plt.show()

    plt.scatter(datosA[:,0], datosA[:,1], c="green", label="Datos con etiqueta -1")
    plt.scatter(datosB[:,0], datosB[:,1], c="red", label="Datos con etiqueta 1")
    plt.contour(x,y,f3([xx,yy]),[0])
    plt.title("Nube uniforme con ruido usando la función 3")
    plt.legend()
    plt.show()

    plt.scatter(datosA[:,0], datosA[:,1], c="green", label="Datos con etiqueta -1")
    plt.scatter(datosB[:,0], datosB[:,1], c="red", label="Datos con etiqueta 1")
    plt.contour(x,y,f4([xx,yy]),[0])
    plt.title("Nube uniforme con ruido usando la función 4")
    plt.legend()
    plt.show()

ej1ap3()

################################################################################
##                             Ejercicio 2                                    ##
################################################################################

#------------------------------------------------------------------------------#
##                              Apartado 1                                    ##
#------------------------------------------------------------------------------#

def ajusta_PLA(datos, label, max_iter, vini):
    w = vini
    for i in range(max_iter):
        w_old = w
        for d,l in zip(datos, label):
            if np.sign(w.dot(d))!=l:
                w = w+l*d
        if np.all(np.equal(w, w_old)):
            return w, i
    return w,max_iter

def evaluaRecta(w, punto):
    return (-w[0]-w[1]*punto)/w[2]

def ej2ap1SinRuido():
    print("Primero sin ruido")
    nube_sin_ruido3d = np.hstack((np.ones(shape=(len(nube_unif),1)),nube_unif))

    w,it = ajusta_PLA(nube_sin_ruido3d, labels_sin_ruido, 10000, np.array([0,0,0]))
    print("Estamos utilizando w_ini = " + str(np.array([0,0,0])))
    print("El número de iteraciones que ha necesitado ha sido: " + str(it))

    nube1 = np.array([nube_sin_ruido3d[i] for i in range(len(nube_sin_ruido3d)) if labels_sin_ruido[i]==1])
    nube2 = np.array([nube_sin_ruido3d[i] for i in range(len(nube_sin_ruido3d)) if labels_sin_ruido[i]==-1])

    plt.scatter(nube1[:,1],nube1[:,2],c="b",label="Clase con etiqueta 1")
    plt.scatter(nube2[:,1],nube2[:,2],c="g",label="Clase con etiqueta -1")
    plt.plot([-51,51],[evaluaRecta(w,-51),evaluaRecta(w,51)],c="r",label="Recta obtenida por PLA con w=[0,0,0]")
    plt.legend()
    plt.show()


    for i in range(10):
        w = np.random.uniform(low=0, high=1, size=3)
        print("Estamos utilizando w_ini = " + str(w))
        w,it = ajusta_PLA(nube_sin_ruido3d, labels_sin_ruido, 10000, w)
        print("El número de iteraciones que ha necesitado ha sido: " + str(it))
        plt.scatter(nube1[:,1],nube1[:,2],c="b",label="Clase con etiqueta 1")
        plt.scatter(nube2[:,1],nube2[:,2],c="g",label="Clase con etiqueta -1")
        plt.plot([-51,51],[evaluaRecta(w,-51),evaluaRecta(w,51)],c="r",label="Recta obtenida por PLA")
        plt.legend()
        plt.show()

def ej2ap1ConRuido():
    print("Ahora con ruido")
    nube_con_ruido3d = np.hstack((np.ones(shape=(len(nube_unif),1)),nube_unif))

    w,it = ajusta_PLA(nube_con_ruido3d, labels_con_ruido, 10000, np.array([0,0,0]))
    print("Estamos utilizando w_ini = " + str(np.array([0,0,0])))
    print("El número de iteraciones que ha necesitado ha sido: " + str(it))

    nube1 = np.array([nube_con_ruido3d[i] for i in range(len(nube_con_ruido3d)) if labels_con_ruido[i]==1])
    nube2 = np.array([nube_con_ruido3d[i] for i in range(len(nube_con_ruido3d)) if labels_con_ruido[i]==-1])

    plt.scatter(nube1[:,1],nube1[:,2],c="b",label="Clase con etiqueta 1")
    plt.scatter(nube2[:,1],nube2[:,2],c="g",label="Clase con etiqueta -1")
    plt.plot([-51,51],[evaluaRecta(w,-51),evaluaRecta(w,51)],c="r",label="Recta obtenida por PLA con w=[0,0,0]")
    plt.legend()
    plt.show()


    for i in range(10):
        w = np.random.uniform(low=0, high=1, size=3)
        print("Estamos utilizando w_ini = " + str(w))
        w,it = ajusta_PLA(nube_con_ruido3d, labels_con_ruido, 10000, w)
        print("El número de iteraciones que ha necesitado ha sido: " + str(it))
        plt.scatter(nube1[:,1],nube1[:,2],c="b",label="Clase con etiqueta 1")
        plt.scatter(nube2[:,1],nube2[:,2],c="g",label="Clase con etiqueta -1")
        plt.plot([-51,51],[evaluaRecta(w,-51),evaluaRecta(w,51)],c="r",label="Recta obtenida por PLA")
        plt.legend()
        plt.show()

ej2ap1SinRuido()
ej2ap1ConRuido()

#------------------------------------------------------------------------------#
##                              Apartado 2                                    ##
#------------------------------------------------------------------------------#

def updateW(X,y,w,minibatch,tasa_aprendizaje):
    '''
    @brief Función dedicada a actualizar el w como se pide en el algoritmo SGD
    @param X conjunto de datos
    @param y conjunto de etiquetas
    @param minibatch conjunto de indices que representan el minibatch
    @param tasa_aprendizaje Tasa de aprendizaje usada en SGD
    '''
    # Obtenemos los datos y etiquetas asociados al minibatch que hemos calculado
    X_minibatch = X[minibatch,:]
    y_minibatch = y[minibatch]
    # Calculamos el factor que vamos a restar a w
    for x,y in zip(X_minibatch, y_minibatch):
        w = w-tasa_aprendizaje*((-y*x)/(1+np.exp(y*w.T.dot(x))))
    return w

def regresionLogisticaSGD(num_epocas_max,X,y,minibatch_size=8,tasa_aprendizaje=0.01, tol=0.01):
    dimension = len(X[0])
    data_size = len(X)
    w = np.zeros(dimension)
    w_old = np.ones(dimension)
    num_epocas = 0
    while num_epocas<num_epocas_max and np.linalg.norm(w_old-w)>=tol:
        w_old=w
        indexes = np.random.choice(data_size, size=data_size, replace=False)
        for i in range(int(data_size/minibatch_size)-1):
            minibatch = indexes[i*minibatch_size:(i+1)*minibatch_size]
            w = updateW(X,y,w,minibatch,tasa_aprendizaje)

        if data_size%minibatch_size!=0:
            resto = (data_size%minibatch_size)*minibatch_size
            minibatch = np.append(indexes[-resto:],indexes[:minibatch_size-resto])
            w = updateW(X,y,w,minibatch,tasa_aprendizaje)
        num_epocas+=1

    return w,num_epocas

def puntoSobreRecta(a,b,punto):
    valor=punto[1]*a+b
    if valor>=punto[2]:
        return True
    else:
        return False

def ej2ap2():
    puntos_uniforme = np.hstack((np.ones(shape=(100,1)),simula_unif(100,2,[0,2])))
    a,b = simula_recta([0,2])
    labels = np.array([0 if puntoSobreRecta(a,b,puntos_uniforme[i]) else 1 for i in range(len(puntos_uniforme))])
    w,iters = regresionLogisticaSGD(10000, puntos_uniforme, labels)

    puntos_uniforme1 = np.array([puntos_uniforme[i] for i in range(len(puntos_uniforme)) if labels[i]==0])
    puntos_uniforme2 = np.array([puntos_uniforme[i] for i in range(len(puntos_uniforme)) if labels[i]==1])

    plt.scatter(puntos_uniforme1[:,1], puntos_uniforme1[:,2],c="blue", label="Puntos con etiqueta 0")
    plt.scatter(puntos_uniforme2[:,1], puntos_uniforme2[:,2],c="green", label="Puntos con etiqueta 1")
    plt.plot([0,2],[b, 2*a+b], c="red", label="Recta frontera")
    plt.legend()
    plt.show()

ej2ap2()
