# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from tabulate import tabulate
from sympy.tensor.array import derive_by_array

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
def gradient_descent(w_init,learning_rate,max_iter,tol,E,symbols,ret_values=False,check_func_value=False):
    w_before=w_init
    w = w_before-learning_rate*gradiente(E,symbols,w_before)
    # Cuidado que no hay el mismo número de elementos que iteraciones, para la primera iteración
    # introducimos dos valores, no uno
    func_values = [evaluate(E,symbols,w_init),evaluate(E,symbols,w)]
    iterations=1
    tolerance_check=np.absolute(evaluate(E,symbols,w)-evaluate(E,symbols,w_before)) if not check_func_value else evaluate(E,symbols,w)
    while iterations<max_iter and tolerance_check>=tol:
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
    eta = 0.01
    maxIter = 10000000000
    error2get = 1e-14
    initial_point = np.array([1.0,1.0])
    u,v = symbols('u v',real=True)
    symbol = [u,v]
    expr = (u**2*exp(v)-2*v**2*exp(-u))**2
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

def Ej1apartado3b():
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
    X = np.matrix(X)
    U,D,VT = np.linalg.svd(X)
    D_mat = np.diag(D)
    xtx_inv = np.transpose(VT)@np.linalg.inv(D_mat)@np.linalg.inv(D_mat)@VT
    pseudo_inverse = xtx_inv@np.transpose(X)
    return np.array(pseudo_inverse.dot(y))[0]

def Error(w,X,y):
    return (1/len(X))*np.sum(np.square(X.dot(w)-y))

def stochasticGradientDescent(max_iter,tasa_aprendizaje,X,y,tol,minibatch_size=64,return_errors=False):
    dimension = len(X[0])
    iter = 0
    w = np.zeros(dimension)
    if return_errors:
        error_hist = [Error(w,X,y)]
    while iter<=max_iter and Error(w,X,y)>=tol:
        minibatch = np.random.choice(len(X), size=minibatch_size, replace=False)
        X_minibatch = X[minibatch,:]
        y_minibatch = y[minibatch]
        substraction = X_minibatch.T.dot(np.dot(X_minibatch,w)-y_minibatch)
        w = w-tasa_aprendizaje*substraction*(2/minibatch_size)
        if return_errors:
            error_hist.append(Error(w,X,y))
        iter+=1
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
    X_train, y_train = readData("./datos/X_train.npy","./datos/y_train.npy")
    X_test, y_test = readData("./datos/X_test.npy","./datos/y_test.npy")

    w_sgd,iter,ein_hist = stochasticGradientDescent(1000,0.01,X_train,y_train,1e-10,return_errors=True)
    w_pseudo = pseudoInversa(X_train,y_train)

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

    plt.scatter(X_train_1[:,1],X_train_1[:,2],c="b",label="Clase con etiqueta -1")
    plt.scatter(X_train_2[:,1],X_train_2[:,2],c="g",label="Clase con etiqueta 1")
    plt.plot([0,1],[-w_sgd[0]/w_sgd[2],(w_sgd[0]-w_sgd[1])/w_sgd[2]],c="r",label="Recta obtenida por SGD")
    plt.plot([0,1],[-w_pseudo[0]/w_pseudo[2],(w_pseudo[0]-w_pseudo[1])/w_pseudo[2]],c="y", label="Recta obtenida por el algoritmo de la pseudo-inversa")
    plt.legend()
    plt.show()

    print("\nEin de SGD: " + str(Error(w_sgd,X_train,y_train)))
    print("Ein de la pseudo-inversa: " + str(Error(w_pseudo,X_train,y_train)))

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
