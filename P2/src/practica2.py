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
    print("#################################\nEjercicio 1, apartado 1\n#################################\n\n")

    # Simulamos datos según una distribución uniforme
    nube_unif = simula_unif(N,2,[-50,50])
    # Simulamos datos según una distribución Gaussiana
    nube_normal = simula_gauss(N,2,[5,7])

    # Pintamos la nube de puntos según una uniforme
    plt.scatter(nube_unif[:,0], nube_unif[:,1])
    plt.title("Nube con la uniforme")
    plt.show()

    input("Presione ENTER para continuar")

    # Pintamos la nube de puntos según una normal
    plt.scatter(nube_normal[:,0], nube_normal[:,1])
    plt.title("Nube con la normal")
    plt.show()

    input("Presione ENTER para continuar")

ej1ap1()

#------------------------------------------------------------------------------#
##                              Apartado 2                                    ##
#------------------------------------------------------------------------------#

def fAp2(x,y,a,b):
    return np.sign(y-a*x-b)

def ej1ap2(N=50):
    print("#################################\nEjercicio 1, apartado 2\n#################################\n\n")
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

    input("Presione ENTER para continuar")

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

    input("Presione ENTER para continuar")

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
    print("#################################\nEjercicio 1, apartado 3\n#################################\n\n")

    # Separamos los datos con ruido del apartado anterior por etiquetas
    datosA = np.array([nube_unif[i] for i in range(len(nube_unif)) if labels_con_ruido[i]==-1])
    datosB = np.array([nube_unif[i] for i in range(len(nube_unif)) if labels_con_ruido[i]==1])

    # Hacemos los meshgrid necesarios para la orden contour para pintar funciones de dos variables
    x = np.arange(-50,50,0.1)
    y = np.arange(-50,50,0.1)
    xx,yy = np.meshgrid(x,y)

    # Dibujamos la función 1
    plt.scatter(datosA[:,0], datosA[:,1], c="green", label="Datos con etiqueta -1")
    plt.scatter(datosB[:,0], datosB[:,1], c="red", label="Datos con etiqueta 1")
    plt.contour(x,y,f1([xx,yy]),[0])
    plt.title("Nube uniforme con ruido usando la función 1")
    plt.legend()
    plt.show()

    input("Presione ENTER para continuar")

    # Dibujamos la función 2
    plt.scatter(datosA[:,0], datosA[:,1], c="green", label="Datos con etiqueta -1")
    plt.scatter(datosB[:,0], datosB[:,1], c="red", label="Datos con etiqueta 1")
    plt.contour(x,y,f2([xx,yy]),[0])
    plt.title("Nube uniforme con ruido usando la función 2")
    plt.legend()
    plt.show()

    input("Presione ENTER para continuar")

    # Dibujamos la función 3
    plt.scatter(datosA[:,0], datosA[:,1], c="green", label="Datos con etiqueta -1")
    plt.scatter(datosB[:,0], datosB[:,1], c="red", label="Datos con etiqueta 1")
    plt.contour(x,y,f3([xx,yy]),[0])
    plt.title("Nube uniforme con ruido usando la función 3")
    plt.legend()
    plt.show()

    input("Presione ENTER para continuar")

    # Dibujamos la función 4
    plt.scatter(datosA[:,0], datosA[:,1], c="green", label="Datos con etiqueta -1")
    plt.scatter(datosB[:,0], datosB[:,1], c="red", label="Datos con etiqueta 1")
    plt.contour(x,y,f4([xx,yy]),[0])
    plt.title("Nube uniforme con ruido usando la función 4")
    plt.legend()
    plt.show()

    input("Presione ENTER para continuar")

ej1ap3()

################################################################################
##                             Ejercicio 2                                    ##
################################################################################

#------------------------------------------------------------------------------#
##                              Apartado 1                                    ##
#------------------------------------------------------------------------------#

def ajusta_PLA(datos, label, max_iter, vini):
    '''
    @brief Método que implementa el algoritmo PLA
    @param datos numpy array con los datos
    @param label etiquetas correspondientes para los datos
    @param max_iter número máximo de iteraciones (épocas)
    @param vini Valor inicial del algoritmo para w
    @return Devuelve el vector de pesos w obtenido que nos da la recta separadora y
    el número de iteraciones (épocas) empleadas
    '''
    # Tomamos el valor inicial
    w = vini
    for i in range(max_iter):
        # Nos quedamos con el w anterior
        w_old = np.copy(w)
        # Para cada dato
        for d,l in zip(datos, label):
            # Si no está bien clasificado entonces actualizamos w
            if np.sign(w.dot(d))!=l:
                w = w+l*d
        # Si w no ha cambiado en una pasada completa de los datos
        if np.all(np.equal(w, w_old)):
            # Devolvemos w y el número de iteraciones
            return w, i
    # Si hemos agotado el número de iteraciones devolvemos lo obenido
    return w,max_iter

def evaluaRecta(w, punto):
    '''
    @brief Función que se encarga de evaluar la recta w en el punto pasado
    @param w numpy array que representa la recta
    @param punto es un número real sobre el que vamos a evaluar la recta
    @return Devuelve un número que es el valor de la recta w en el punto
    pasado como parámetro
    '''
    return (-w[0]-w[1]*punto)/w[2]

def ej2ap1SinRuido():
    print("#################################\nEjercicio 2, apartado 1 sin ruido\n#################################\n\n")
    # Tomamos los datos sin ruido del ejercicio 1 y le colocamos 1s al principio
    nube_sin_ruido3d = np.hstack((np.ones(shape=(len(nube_unif),1)),nube_unif))

    # Ejecutamos el algoritmo PLA sobre los datos con las etiquetas sin ruido empezando en [0,0,0]
    w,it = ajusta_PLA(nube_sin_ruido3d, labels_sin_ruido, 10000, np.array([0,0,0]))
    print("Estamos utilizando w_ini = " + str(np.array([0,0,0])))
    print("El número de iteraciones que ha necesitado ha sido: " + str(it))

    # Dividimos los datos por etiquetas
    nube1 = np.array([nube_sin_ruido3d[i] for i in range(len(nube_sin_ruido3d)) if labels_sin_ruido[i]==1])
    nube2 = np.array([nube_sin_ruido3d[i] for i in range(len(nube_sin_ruido3d)) if labels_sin_ruido[i]==-1])

    # Imprimimos la recta obtenida con PLA empezando en [0,0,0]
    plt.scatter(nube1[:,1],nube1[:,2],c="b",label="Clase con etiqueta 1")
    plt.scatter(nube2[:,1],nube2[:,2],c="g",label="Clase con etiqueta -1")
    plt.plot([-51,51],[evaluaRecta(w,-51),evaluaRecta(w,51)],c="r",label="Recta obtenida por PLA sin ruido con w=[0,0,0]")
    plt.legend()
    plt.show()

    input("Presione ENTER para continuar")

    # Generamos 10 vectores inciales aleatorios con valores entre 0 y 1
    for i in range(10):
        w = np.random.uniform(low=0, high=1, size=3)
        print("\nEstamos utilizando w_ini = " + str(w))
        # Obtenemos la w con la inicial aleatoria y los datos sin ruido
        w,it = ajusta_PLA(nube_sin_ruido3d, labels_sin_ruido, 10000, w)
        print("El número de iteraciones que ha necesitado ha sido: " + str(it))
        # Pintamos la recta obtenida
        plt.scatter(nube1[:,1],nube1[:,2],c="b",label="Clase con etiqueta 1")
        plt.scatter(nube2[:,1],nube2[:,2],c="g",label="Clase con etiqueta -1")
        plt.plot([-51,51],[evaluaRecta(w,-51),evaluaRecta(w,51)],c="r",label="Recta obtenida por PLA sin ruido " + str(i))
        plt.legend()
        plt.show()
        input("Presione ENTER para continuar")

def ej2ap1ConRuido():
    print("#################################\nEjercicio 2, apartado 1 con ruido\n#################################\n\n")
    # Tomamos los datos con ruido del ejercicio 1 y le colocamos 1s al principio
    nube_con_ruido3d = np.hstack((np.ones(shape=(len(nube_unif),1)),nube_unif))

    # Ejecutamos el algoritmo PLA sobre los datos con las etiquetas con ruido empezando en [0,0,0]
    w,it = ajusta_PLA(nube_con_ruido3d, labels_con_ruido, 10000, np.array([0,0,0]))
    print("Estamos utilizando w_ini = " + str(np.array([0,0,0])))
    print("El número de iteraciones que ha necesitado ha sido: " + str(it))

    # Dividimos los datos por etiquetas
    nube1 = np.array([nube_con_ruido3d[i] for i in range(len(nube_con_ruido3d)) if labels_con_ruido[i]==1])
    nube2 = np.array([nube_con_ruido3d[i] for i in range(len(nube_con_ruido3d)) if labels_con_ruido[i]==-1])

    # Imprimimos la recta obtenida con PLA empezando en [0,0,0]
    plt.scatter(nube1[:,1],nube1[:,2],c="b",label="Clase con etiqueta 1")
    plt.scatter(nube2[:,1],nube2[:,2],c="g",label="Clase con etiqueta -1")
    plt.plot([-51,51],[evaluaRecta(w,-51),evaluaRecta(w,51)],c="r",label="Recta obtenida por PLA con ruido con w=[0,0,0]")
    plt.legend()
    plt.show()

    input("Presione ENTER para continuar")

    # Generamos 10 vectores inciales aleatorios con valores entre 0 y 1
    for i in range(10):
        w = np.random.uniform(low=0, high=1, size=3)
        print("\nEstamos utilizando w_ini = " + str(w))
        # Obtenemos la w con la inicial aleatoria y los datos con ruido
        w,it = ajusta_PLA(nube_con_ruido3d, labels_con_ruido, 10000, w)
        print("El número de iteraciones que ha necesitado ha sido: " + str(it))
        # Pintamos la recta obtenida
        plt.scatter(nube1[:,1],nube1[:,2],c="b",label="Clase con etiqueta 1")
        plt.scatter(nube2[:,1],nube2[:,2],c="g",label="Clase con etiqueta -1")
        plt.plot([-51,51],[evaluaRecta(w,-51),evaluaRecta(w,51)],c="r",label="Recta obtenida por PLA con ruido " + str(i))
        plt.legend()
        plt.show()
        input("Presione ENTER para continuar")

ej2ap1SinRuido()
ej2ap1ConRuido()

#------------------------------------------------------------------------------#
##                              Apartado 2                                    ##
#------------------------------------------------------------------------------#

def regresionLogisticaSGD(num_epocas_max,X,y,tasa_aprendizaje=0.01, tol=0.01):
    '''
    @brief Función que implementa el algoritmo SGD orientado a regresión logística
    @param num_epocas_max Número máximo de épocas
    @param X numpy array con los datos
    @param y numpy array con las etiquetas de los datos
    @param tasa_aprendizaje Tasa de aprendizaje para SGD
    @param tol Tolerancia que gestiona la parada del algoritmo
    '''
    # Obtenemos la dimensión del conjuntos de datos
    dimension = len(X[0])
    data_size = len(X)
    # Empezamos en w=[0,0,0]
    w = np.zeros(dimension)
    # Ponemos al principio w_old como [1,1,1]
    w_old = np.ones(dimension)
    # Inicializamos el número de épocas a 0
    num_epocas = 0
    # Mientras que no superemos el número máximo de épocas y que la diferencia entre w y w_old sea mayor que la tolerancia
    while num_epocas<num_epocas_max and np.linalg.norm(w_old-w)>=tol:
        # Actualizamos w_old
        w_old=np.copy(w)
        # Barajamos los índices
        indexes = np.random.choice(data_size, size=data_size, replace=False)
        # Para cada índice actualizamos w
        for id in indexes:
            w = w-tasa_aprendizaje*((-y[id]*X[id])/(1+np.exp(y[id]*w.dot(X[id]))))
        # Sumamos 1 al número de épocas
        num_epocas+=1

    # Devolvemos el número de épocas usadas y el w obtenido
    return w,num_epocas

def puntoSobreRecta(a,b,punto):
    '''
    @brief Función que determina si un punto está sobre la recta dada por y=a*x+b
    @param a Pendiente de la recta
    @param b Término independiente de la recta
    @param punto punto de la forma [1,x,y] sobre el que queremos valorar si está o no
    por encima de la recta
    @return Devuelve True si el punto está por encima de la recta o en la recta y
    False si está por debajo.
    '''
    # Obtenemos el valor de la recta en el punto x
    valor=punto[1]*a+b
    # Si el valor de la recta es menor que el del punto entonces el punto está por encima
    if valor<punto[2]:
        return True
    else:
        return False

def Eout(datos, labels, w):
    '''
    @brief Función que calcula el error cometido por w en los datos pasados.
    @param datos Datos sobre los que se va a calcular el error
    @param labels Etiquetas reales de los datos pasados
    @param w Recta separadora obtenida
    @return Devuelve el erorr
    '''
    # Calculamos el error según las transparencias de teoría, esto es
    # 1/N * sum(ln(1+e^(-y_n*w^T*x_n)))
    tam_datos = len(datos)
    error = 0
    for d,l in zip(datos,labels):
        error+=np.log(1+np.exp(-l*w.dot(d)))
    return error/tam_datos

def ej2ap2():
    print("#################################\nEjercicio 2, apartado 2\n#################################\n\n")
    # Simulamos 100 datos y colocamos 1s delante
    puntos_uniforme = np.hstack((np.ones(shape=(100,1)),simula_unif(100,2,[0,2])))
    # Creamos una recta aleatoria
    a,b = simula_recta([0,2])
    # Calculamos las etiquetas en función de si los puntos quedan o no por encima de la recta aleatoria
    labels = np.array([1 if puntoSobreRecta(a,b,puntos_uniforme[i]) else -1 for i in range(len(puntos_uniforme))])
    # Calculamos la estimación de w mediante SGD aplicado a regresión logística
    w,iters = regresionLogisticaSGD(10000, puntos_uniforme, labels)

    print("El algoritmo ha convergido en " + str(iters) + " iteraciones")

    # Calculamos el Ein
    error = Eout(puntos_uniforme, labels, w)
    print("El error dentro de la muestra es: " + str(error))

    # Dividimos los datos por clases
    puntos_uniforme1 = np.array([puntos_uniforme[i] for i in range(len(puntos_uniforme)) if labels[i]==1])
    puntos_uniforme2 = np.array([puntos_uniforme[i] for i in range(len(puntos_uniforme)) if labels[i]==-1])

    # Pintamos los datos con la recta aleatoria de frontera para ver que lo estamos haciendo bien y cuál sería una solución óptima
    plt.scatter(puntos_uniforme1[:,1], puntos_uniforme1[:,2],c="blue", label="Puntos con etiqueta 1")
    plt.scatter(puntos_uniforme2[:,1], puntos_uniforme2[:,2],c="green", label="Puntos con etiqueta -1")
    plt.plot([0,2],[b, 2*a+b], c="red", label="Recta frontera")
    plt.legend()
    plt.show()

    input("Presione ENTER para continuar")

    # Pintamos la estimación de RL SGD con los 100 datos
    plt.scatter(puntos_uniforme1[:,1], puntos_uniforme1[:,2],c="blue", label="Puntos con etiqueta 1")
    plt.scatter(puntos_uniforme2[:,1], puntos_uniforme2[:,2],c="green", label="Puntos con etiqueta -1")
    plt.plot([0,2],[evaluaRecta(w,0), evaluaRecta(w,2)], c="red", label="Recta Regresión Logística SGD")
    plt.legend()
    plt.show()

    input("Presione ENTER para continuar")

    # Para el tamaño del conjunto desde 1000 a 5000
    for num_puntos in [1000,2000,3000,4000,5000]:
        # Calculamos una muestra de num_puntos de tamaño
        puntos_uniforme = np.hstack((np.ones(shape=(num_puntos,1)),simula_unif(num_puntos,2,[0,2])))
        # Calculamos sus etiquetas
        labels = np.array([1 if puntoSobreRecta(a,b,puntos_uniforme[i]) else -1 for i in range(len(puntos_uniforme))])

        # Calculamos el error cometido
        error = Eout(puntos_uniforme, labels, w)
        print("El error con " + str(num_puntos) + " datos ha sido: " + str(error))

        # Separamos los datos por clases
        puntos_uniforme1 = np.array([puntos_uniforme[i] for i in range(len(puntos_uniforme)) if labels[i]==1])
        puntos_uniforme2 = np.array([puntos_uniforme[i] for i in range(len(puntos_uniforme)) if labels[i]==-1])

        # Pintamos w para el nuevo conjunto de datos generado
        plt.scatter(puntos_uniforme1[:,1], puntos_uniforme1[:,2],c="blue", label="Puntos con etiqueta 1")
        plt.scatter(puntos_uniforme2[:,1], puntos_uniforme2[:,2],c="green", label="Puntos con etiqueta -1")
        plt.plot([0,2],[evaluaRecta(w,0), evaluaRecta(w,2)], c="red", label="Recta Regresión Logística SGD")
        plt.legend()
        plt.show()
        input("Presione ENTER para continuar")

ej2ap2()
