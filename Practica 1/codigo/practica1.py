# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sympy import *                     # Para la derivación de funciones
from tabulate import tabulate           # Para colocar los datos como una tabla al hacer print

# Fijamos la semilla aleatoria
np.random.seed(1)

################################################################################
##                             Ejercicio 1                                    ##
################################################################################

#------------------------------------------------------------------------------#
##                              Apartado 1                                    ##
#------------------------------------------------------------------------------#
def evaluate(E,symbols,w):
    '''
    @brief función que se encarga de evaluar la función dada por la expresión E
    que está formada por las variables symbols y que se quiere evaluar en el punto w.
    @param E expresión escrita en sintaxis de SymPy que se quiere evaluar
    @param symbols lista de símbolos de los que se compone la expresión E
    @param w punto (en forma de lista o numpy array) en el que se quiere evaluar
    la expresión E
    @return Devuelve un valor numérico que representa el valor de la función codificada
    por la expresión E en el punto w.
    '''
    # Hacemos un diccionario de la forma {simbolo1:w[0],...,simbolon:w[n]}
    point = dict(zip(symbols,w))
    # Devolvemos la evaluación de la función
    return N(E.subs(point))

def gradiente(E,symbols,w):
    '''
    @brief Función que obtiene el gradiente de la función dada por la expresión E
    y la evalúa en el punto w
    @param E Expresión de la que se quiere obtener el gradiente dada en sintaxis
    de SymPy
    @param symbols Símbolos de los que depende la función dada por la expresión E
    dados como una lista
    @param w Punto en el que queremos que se evalúe el vector gradiente dado como una lista
    de python o como un numpy array
    @return Devuelve un vector que representa el gradiente de la función dada por
    la expresión E en función de las variables que se incluyen dentro de la lista symbols
    evaluado en el punto w.
    '''
    gradiente = []
    # Obtenemos la derivada para cada una de las variables, la evaluamos en el punto y
    # ponemos el valor en el vector gradiente
    for i in range(len(symbols)):
        dife = diff(E,symbols[i])
        gradiente.append(evaluate(dife,symbols,w))
    # Lo devolvemos como un numpy array
    return np.array(gradiente)

def gradient_descent(w_init,learning_rate,max_iter,tol,E,symbols,ret_values=False,check_func_value=False):
    '''
    @brief Función que implementa el algoritmo de gradiente descendente orientado
    a encontrar el mínimo de una función
    @param w_init Punto inicial del que se parten los cálculos
    @param learning_rate Factor llamado tasa de aprendizaje que se utiliza para
    ajustar el valor w en la siguiente iteración
    @param max_iter Número de iteraciones máximas dadas para la ejecución del algoritmo
    @param tol Tolerancia mínima, esto es, una condición de que el algoritmo ha obtenido una solución
    considerada como aceptable en términos del error
    @param E Expresión de la función que queremos minimizar
    @param symbols Símbolos de los que depende la función (es una lista)
    @param ret_values Es una condición booleana que si es falsa sólo se devuelven
    el número de iteraciones consumidas y el w final obtenido. En caso de ser verdadera
    se devuelve además una lista con los valores de la función para cada paso de w calculado
    @param check_func_value Condición booleana que, de ser verdadera, se utliza como tolerancia
    el valor de la función en el w actual. Si es falsa se utiliza la diferencia de las imágenes
    del w actual con el anterior como medida de tolerancia.
    @return Se devuelve el punto mínimo, el número de iteraciones empleado y si ret_values es verdadero
    una lista con los valores que ha ido tomando la función para cada w calculado.
    '''
    # Hacemos el primer cálculo de w
    w_before=w_init
    w = w_before-learning_rate*gradiente(E,symbols,w_before)
    # Cuidado que no hay el mismo número de elementos que iteraciones, para la primera iteración
    # introducimos dos valores, no uno
    func_values = [evaluate(E,symbols,w_init),evaluate(E,symbols,w)]
    iterations=1
    # Actualizamos la condición de la tolerancia
    tolerance_check=np.absolute(evaluate(E,symbols,w)-evaluate(E,symbols,w_before)) if not check_func_value else evaluate(E,symbols,w)
    while iterations<max_iter and tolerance_check>=tol:
        # Mantenemos el w actual y el w anterior para poder calcular la tolerancia
        w_before=w
        w = w_before-learning_rate*gradiente(E,symbols,w_before)
        func_values.append(evaluate(E,symbols,w))
        iterations+=1
        tolerance_check=np.absolute(evaluate(E,symbols,w)-evaluate(E,symbols,w_before)) if not check_func_value else evaluate(E,symbols,w)
    if ret_values:
        return w, iterations, func_values
    else:
        return w, iterations

#------------------------------------------------------------------------------#
##                               Apartado 2                                   ##
#------------------------------------------------------------------------------#

def Ej1apartado2():
    # Definimos los parámetros para el algoritmo de gradiente descendente como se indica en el enunciado
    eta = 0.01
    maxIter = 10000000000
    error2get = 1e-14
    initial_point = np.array([1.0,1.0])
    # Definimos los simbolos de la función y la expresión de la misma.
    u,v = symbols('u v',real=True)
    symbol = [u,v]
    expr = (u**2*exp(v)-2*v**2*exp(-u))**2
    # Obtenemos el resultado de gradiente descendente para los parámetros anteriores
    w, it = gradient_descent(initial_point,eta,maxIter,error2get,expr,symbol,check_func_value=True)
    print("#################################\nEjercicio 1, apartado 2\n#################################\n\n")
    print("El número de iteraciones empleado ha sido de: " + str(it))
    print("El mínimo encontrado por gradiente desdendente ha sido: ")
    print("x: " + str(w[0]))
    print("y: " + str(w[1]))
    input("Presione ENTER para continuar")

#------------------------------------------------------------------------------#
##                               Apartado 3                                   ##
#------------------------------------------------------------------------------#

def Ej1apartado3a(eta):
    # Definimos los parámetros para el algoritmo de gradiente descendente como se indica en el enunciado
    maxIter = 50
    error2get = 1e-14
    initial_point = np.array([0.1,0.1])
    # Definimos los simbolos de la función y la expresión de la misma.
    x,y = symbols('x y',real=True)
    symbol = [x,y]
    expr = x**2 + 2*y**2 + 2*sin(2*pi*x)*sin(2*pi*y)
    # Obtenemos el resultado de gradiente descendente para los parámetros anteriores
    w, it, values = gradient_descent(initial_point,eta,maxIter,error2get,expr,symbol,ret_values=True)
    print("#################################\nEjercicio 1, apartado 3 a\n#################################\n\n")
    print("Estamos utilizando eta = " + str(eta))
    print("El número de iteraciones empleado ha sido de: " + str(it))
    print("El mínimo encontrado por gradiente desdendente ha sido: ")
    print("x: " + str(w[0]))
    print("y: " + str(w[1]))

    # Generamos un gráfico de la evolución de los valores de la función dada por expr
    # a cada paso del algoritmo gradiente descendente
    plt.plot(list(range(len(values))),values,label="Valores de la función para cada iteración")
    plt.legend()
    plt.show()
    input("Presione ENTER para continuar")

def Ej1apartado3b():
    # Definimos los parámetros para el algoritmo de gradiente descendente como se indica en el enunciado
    eta=0.01
    maxIter = 50
    error2get = 1e-14
    # Definimos los simbolos de la función y la expresión de la misma.
    x,y = symbols('x y',real=True)
    symbol = [x,y]
    expr = x**2 + 2*y**2 + 2*sin(2*pi*x)*sin(2*pi*y)
    ws_init = [np.array([0.1,0.1]),np.array([1,1]),np.array([-0.5,-0.5]),np.array([-1,-1])]
    results = []
    # Para cada uno de los puntos iniciales obtenemos el mínimo por gradiente descendente
    for w_init in ws_init:
        w, it = gradient_descent(w_init,eta,maxIter,error2get,expr,symbol)
        # Añadimos en la lista resuts el punto incial, el punto mínimo obtenido y su valor de la función.
        results.append([w_init,w,evaluate(expr,symbol,w)])

    print("#################################\nEjercicio 1, apartado 3 b\n#################################\n\n")
    print("Tabla con los valores de los mínimos encontrados: \n")
    # Uso la librería tabulate para darle formato de tabla a la salida
    print(tabulate(results,headers=["Punto inicial", "Mínimo encontrado", "Valor en el mínimo"]))
    input("Presione ENTER para continuar")


Ej1apartado2()
Ej1apartado3a(0.01)
Ej1apartado3a(0.1)
Ej1apartado3b()


################################################################################
##                               Ejercicio 2                                  ##
################################################################################

#------------------------------------------------------------------------------#
##                               Apartado 1                                   ##
#------------------------------------------------------------------------------#

def pseudoInversa(X,y):
    '''
    @brief Función que implementa el algoritmo de la pseudo inversa
    @param X datos (en formato numpy array o numpy matrix) de los que queremos obtener
    la función lineal que los separa
    @param y etiquetas de los datos de X (en formato numpy array)
    @return Devuelve un numpy array que representa a w (los coeficientes de la función lineal asociada)
    '''
    # Convertimos los datos a formato matriz
    X = np.matrix(X)
    # Calculamos la descomposición en valores singulares
    U,D,VT = np.linalg.svd(X)
    # Creamos una matriz diagonal a partir de los valores de D (que es un array)
    D_mat = np.diag(D)
    # Calculamos la pseudo inversa de X^TX
    xtx_inv = np.transpose(VT)@np.linalg.inv(D_mat)@np.linalg.inv(D_mat)@VT
    # Calculamos la matriz (X^TX)^-1X^T
    pseudo_inverse = xtx_inv@np.transpose(X)
    # Devolvemos los coeficientes de la función lineal asociada que separa los datos
    return np.array(pseudo_inverse.dot(y))[0]

def Error(w,X,y):
    '''
    @brief Función que calcula el error de w al dividir los datos de X con etiquetas y
    como función lineal
    @param w coeficientes de la función lineal asociada a la división de los datos de X
    con etiquetas y
    @param X datos
    @param y etiquetas
    '''
    return (1/len(X))*np.sum(np.square(X.dot(w)-y))

def stochasticGradientDescent(max_iter,tasa_aprendizaje,X,y,tol,minibatch_size=64,return_errors=False):
    '''
    @brief Función que implementa el algoritmo gradiente descendente estocástico
    @param max_iter Número máximo de iteraciones a emplear
    @param tasa_aprendizaje Tasa de aprendizaje que multiplica al factor con el que se actualiza w
    @param X Conjunto de datos de entrenamiento
    @param y etiquetas de los datos de entrenamiento
    @param tol Tolerancia asociada al error, se considera una solución admisible cuando
    el error es menor que esta tolerancia
    @param minibatch_size Tamaño del minibatch, por defecto 64
    @param return_errors Condición booleana, que de ser cierta hace que la función devuelva
    un vector con la evolución del error a cada iteración
    @return Devuelve el w obtenido que ajusta la función lineal que divide los datos, el número
    de iteraciones consumidad y, en caso de ser return_errors verdadero, una lista de errores a lo largo
    de las iteraciones del algoritmo.
    '''
    # Obtenemos la dimensión de los datos
    dimension = len(X[0])
    iter = 0
    # Inicializamos el vector w inicial a ceros
    w = np.zeros(dimension)
    # Si se requieren los errores los computamos
    if return_errors:
        error_hist = [Error(w,X,y)]
    # Hasta que agotemos las iteraciones máximas o el error sea menor que la tolerancia
    while iter<=max_iter and Error(w,X,y)>=tol:
        # Calculamos un conjunto de índices de tamaño minibatch_size que serán nuestro minibatch
        minibatch = np.random.choice(len(X), size=minibatch_size, replace=False)
        # Obtenemos los datos y etiquetas asociados al minibatch que hemos calculado
        X_minibatch = X[minibatch,:]
        y_minibatch = y[minibatch]
        # Calculamos el factor que vamos a restar a w
        substraction = X_minibatch.T.dot(np.dot(X_minibatch,w)-y_minibatch)
        # Actualizamos el valor de w multiplicando por la tasa de aprendizaje y como indican las
        # transparencias de teoría
        w = w-tasa_aprendizaje*substraction*(2/minibatch_size)
        if return_errors:
            error_hist.append(Error(w,X,y))
        iter+=1
    # Devolvemos w, el número de iteraciones y los errores si es necesario
    if not return_errors:
        return w,iter
    else:
        return w,iter,error_hist

# Funcion para leer los datos
def readData(file_x, file_y):
    label5 = 1
    label1 = -1
	# Leemos los ficheros
    datax = np.load(file_x)
    datay = np.load(file_y)
    y = []
    x = []
	# Solo guardamos los datos cuya clase sea la 1 o la 5
    for i in range(0,datay.size):
        if datay[i] == 5 or datay[i] == 1:
            if datay[i] == 5:
                y.append(label5)
            else:
                y.append(label1)
            x.append(np.array([1, datax[i][0], datax[i][1]]))

    x = np.array(x, np.float64)
    y = np.array(y, np.float64)

    return x, y

def Ej2apartado1():
    # Leemos los conjuntos de train y test
    X_train, y_train = readData("./datos/X_train.npy","./datos/y_train.npy")
    X_test, y_test = readData("./datos/X_test.npy","./datos/y_test.npy")

    # Calculamos w con gradiente descendente estocastico y con el algoritmo de la pseudo inversa
    w_sgd,iter,ein_hist = stochasticGradientDescent(1000,0.01,X_train,y_train,1e-10,return_errors=True)
    w_pseudo = pseudoInversa(X_train,y_train)

    # Separamos el conjunto de train por clases para poder hacer el plot de forma visual por colores
    X_train_1 = []
    X_train_2 = []

    for i in range(len(X_train)):
        if y_train[i]==-1:
            X_train_1.append(X_train[i])
        else:
            X_train_2.append(X_train[i])

    X_train_1 = np.array(X_train_1)
    X_train_2 = np.array(X_train_2)

    print("\n\n#################################\nEjercicio 2, apartado 1\n#################################\n\n")

    print("W SGD: " + str(w_sgd))
    print("W Pseudo: " + str(w_pseudo))

    # Hacemos una gráfica con los datos separados por clases y las dos rectas obtenidas
    plt.scatter(X_train_1[:,1],X_train_1[:,2],c="b",label="Clase con etiqueta -1")
    plt.scatter(X_train_2[:,1],X_train_2[:,2],c="g",label="Clase con etiqueta 1")
    plt.plot([0,1],[-w_sgd[0]/w_sgd[2],(w_sgd[0]-w_sgd[1])/w_sgd[2]],c="r",label="Recta obtenida por SGD")
    plt.plot([0,1],[-w_pseudo[0]/w_pseudo[2],(w_pseudo[0]-w_pseudo[1])/w_pseudo[2]],c="y", label="Recta obtenida por el algoritmo de la pseudo-inversa")
    plt.legend()
    plt.show()

    print("\nEin de SGD: " + str(Error(w_sgd,X_train,y_train)))
    print("Ein de la pseudo-inversa: " + str(Error(w_pseudo,X_train,y_train)))

    # Hacemos un plot de la evolución de los errores
    plt.plot(list(range(len(ein_hist))),ein_hist,label="Evolución de Ein")
    plt.legend()
    plt.show()

    print("\nEout de SGD: " + str(Error(w_sgd,X_test,y_test)))
    print("Eout de la pseudo-inversa: " + str(Error(w_pseudo,X_test,y_test)))
    input("Presione ENTER para continuar")

Ej2apartado1()

#------------------------------------------------------------------------------#
##                               Apartado 2                                   ##
#------------------------------------------------------------------------------#

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

def f(x,y):
    '''
    @brief Función que implementa la función con la que vamos a trabajar
    @param x Argumento de la función
    @param y Argumento de la función
    @return Valor numérico de la función del enunciado
    '''
    return np.sign((x-0.2)**2 + y**2 - 0.6)

def Ej2apartado2(niter=1000):
    print("#################################\nEjercicio 2, apartado 2\n#################################\n\n")
    muestra = simula_unif(1000,2,1)
    plt.scatter(muestra[:,0],muestra[:,1],label="Muestra de 1000 puntos según una uniforme")
    plt.legend()
    plt.show()

    labels = np.array([f(x,y) for x,y in muestra],dtype=np.float64)

    muestra_no_noise_lab1 = np.array([muestra[i] for i in range(len(labels)) if labels[i]==1])
    muestra_no_noise_lab2 = np.array([muestra[i] for i in range(len(labels)) if labels[i]==-1])
    plt.scatter(muestra_no_noise_lab1[:,0],muestra_no_noise_lab1[:,1],label="Clase con etiqueta 1")
    plt.scatter(muestra_no_noise_lab2[:,0],muestra_no_noise_lab2[:,1],label="Clase con etiqueta -1")
    plt.title("Antes del ruido")
    plt.legend()
    plt.show()

    ind_noise = np.random.choice(len(labels),int(0.1*len(labels)),replace=False)
    labels[ind_noise] = -labels[ind_noise]

    muestra_lab1 = np.array([muestra[i] for i in range(len(labels)) if labels[i]==1])
    muestra_lab2 = np.array([muestra[i] for i in range(len(labels)) if labels[i]==-1])

    plt.scatter(muestra_lab1[:,0],muestra_lab1[:,1],label="Clase con etiqueta 1")
    plt.scatter(muestra_lab2[:,0],muestra_lab2[:,1],label="clase con etiqueta -1")
    plt.title("Después de introducir ruido")
    plt.legend()
    plt.show()

    vec_caract = np.hstack((np.ones(shape=(1000,1)),muestra))
    w,it = stochasticGradientDescent(50,0.01,vec_caract,labels,1e-10)
    print("El error obtenido (Ein): " + str(Error(w,vec_caract,labels)))
    vec_caract_out = np.hstack((np.ones(shape=(1000,1)),simula_unif(1000,2,1)))
    labels_out = np.array([f(y,z) for x,y,z in vec_caract_out],dtype=np.float64)
    print("El error obtenido (Eout): " + str(Error(w,vec_caract_out,labels_out)))

    plt.scatter(muestra_lab1[:,0],muestra_lab1[:,1],c="b",label="Clase con etiqueta -1")
    plt.scatter(muestra_lab2[:,0],muestra_lab2[:,1],c="g",label="Clase con etiqueta 1")
    plt.plot([0,1],[-w[0]/w[2],(w[0]-w[1])/w[2]],c="r",label="Recta obtenida por SGD")
    plt.axis((-1,1,-1,1))
    plt.legend()
    plt.show()

    hist_ein = np.array([])
    hist_eout = np.array([])
    for i in range(niter):
        vec_caract = np.hstack((np.ones(shape=(1000,1)),simula_unif(1000,2,1)))
        labels = np.array([f(y,z) for x,y,z in vec_caract],dtype=np.float64)
        ind_noise = np.random.choice(len(labels),int(0.1*len(labels)),replace=False)
        labels[ind_noise] = -labels[ind_noise]
        w,it = stochasticGradientDescent(50,0.01,vec_caract,labels,1e-10)
        hist_ein = np.append(hist_ein,Error(w,vec_caract,labels))
        muestra_out = np.hstack((np.ones(shape=(1000,1)),simula_unif(1000,2,1)))
        labels_out = np.array([f(y,z) for x,y,z in muestra_out],dtype=np.float64)
        hist_eout = np.append(hist_eout,Error(w,muestra_out,labels_out))
    print("Media Ein: " + str(np.mean(hist_ein)))
    print("Media Eout: " + str(np.mean(hist_eout)))
    input("Presione ENTER para continuar")

Ej2apartado2()

################################################################################
##                                  Bonus                                     ##
################################################################################

def hessian(E,x,y,w):
    return np.array([[N(E.diff(x).diff(x).subs({x:w[0],y:w[1]})),N(E.diff(x).diff(y).subs({x:w[0],y:w[1]}))],[N(E.diff(y).diff(x).subs({x:w[0],y:w[1]})),N(E.diff(y).diff(y).subs({x:w[0],y:w[1]}))]],dtype=np.float64)

def newton(max_iter,tol,w_init,E,x,y,step=0.01):
    iter = 0
    w=w_init
    hist_values = [evaluate(E,[x,y],w)]
    while iter<=max_iter and evaluate(E,[x,y],w):
        w = w - step*np.linalg.inv(hessian(E,x,y,w)).dot(gradiente(E,[x,y],w))
        hist_values.append(evaluate(E,[x,y],w))
        iter+=1
    return w,iter,hist_values

def bonus():
    print("\n\n#################################\nBonus\n#################################\n\n")
    x,y = symbols('x y',real=True)
    expr = x**2 + 2*y**2 + 2*sin(2*pi*x)*sin(2*pi*y)
    ws_init = [np.array([0.1,0.1]),np.array([1,1]),np.array([-0.5,-0.5]),np.array([-1,-1])]
    for w_init in ws_init:
        w,iter,hist_values = newton(50,1e-10,w_init,expr,x,y)
        print("w: " + str(w) + " valor: " + str(evaluate(expr,[x,y],w)))
        plt.plot(list(range(len(hist_values))),hist_values,label="Punto inicial " + str(w_init))
        plt.legend()
        plt.show()
    input("Presione ENTER para continuar")

bonus()
