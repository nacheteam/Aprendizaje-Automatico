# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split

# Inicializamos la semilla
np.random.seed(123456789)

################################################################################
##                           FUNCIONES AUXILIARES                             ##
################################################################################

def readData(path="./datos/optdigits"):
    '''
    @brief Función que lee los ficheros de test y train de optdigits y los unifica en
    una matriz de datos y un vector de labels
    @param path Carpeta en la que están los datos
    '''
    # Abrimos los ficheros de test y train
    test = open(path+"/optdigits.tes", "r")
    train = open(path+"/optdigits.tra", "r")

    #Inicializamos el dataset y las etiquetas
    dataset = []
    labels = []
    # Para cada linea de los ficheros, separamos los elementos por comas y obtenemos
    # el vector de características y las etiquetas
    for line in test:
        row=[]
        for el in line.split(",")[:-1]:
            row.append(float(el))
        dataset.append(row)
        labels.append(int(line.split(",")[-1]))
    for line in train:
        row=[]
        for el in line.split(",")[:-1]:
            row.append(float(el))
        dataset.append(row)
        labels.append(int(line.split(",")[-1]))
    return np.array(dataset), np.array(labels)

################################################################################
##                 FUNCIONES DE PRUEBA DE ALGORITMOS                          ##
################################################################################

def pruebaMinimosCuadradosRL(train_data, test_data, train_labels, test_labels):
    '''
    @brief Función que aplica mínimos cuadrados y obtiene la valoración del ajuste.
    @param train_data Datos de entrenamiento
    @param test_data Datos de test
    @param train_labels Etiquetas de los datos de entrenamiento
    @param test_labels Etiquetas de los datos de test
    @return Devuelve el score del ajuste con los datos de entrenamiento valorados
    con los de test
    '''
    reg = linear_model.LinearRegression().fit(train_data,train_labels)
    return reg.score(test_data,test_labels)

def pruebaRidge(train_data, test_data, train_labels, test_labels):
    '''
    @brief Función que aplica el modelo de Ridge y obtiene la valoración del ajuste.
    @param train_data Datos de entrenamiento
    @param test_data Datos de test
    @param train_labels Etiquetas de los datos de entrenamiento
    @param test_labels Etiquetas de los datos de test
    @return Devuelve el score del ajuste con los datos de entrenamiento valorados
    con los de test
    '''
    clf=linear_model.Ridge(alpha=0.1)
    clf.fit(train_data, train_labels)
    return clf.score(test_data, test_labels)

def pruebaLasso(train_data, test_data, train_labels, test_labels):
    '''
    @brief Función que aplica LASSO y obtiene la valoración del ajuste.
    @param train_data Datos de entrenamiento
    @param test_data Datos de test
    @param train_labels Etiquetas de los datos de entrenamiento
    @param test_labels Etiquetas de los datos de test
    @return Devuelve el score del ajuste con los datos de entrenamiento valorados
    con los de test
    '''
    reg = linear_model.Lasso(alpha=0.1)
    reg.fit(train_data, train_labels)
    return reg.score(test_data, test_labels)

def pruebaLARSLasso(train_data, test_data, train_labels, test_labels):
    '''
    @brief Función que aplica LARS Lasso y obtiene la valoración del ajuste.
    @param train_data Datos de entrenamiento
    @param test_data Datos de test
    @param train_labels Etiquetas de los datos de entrenamiento
    @param test_labels Etiquetas de los datos de test
    @return Devuelve el score del ajuste con los datos de entrenamiento valorados
    con los de test
    '''
    reg = linear_model.LassoLars(alpha=0.1)
    reg.fit(train_data, train_labels)
    return reg.score(test_data, test_labels)

def pruebaSGDClassifier(train_data, test_data, train_labels, test_labels):
    '''
    @brief Función que aplica Gradiente Descendente Estocástico y obtiene la
    valoración del ajuste.
    @param train_data Datos de entrenamiento
    @param test_data Datos de test
    @param train_labels Etiquetas de los datos de entrenamiento
    @param test_labels Etiquetas de los datos de test
    @return Devuelve el score del ajuste con los datos de entrenamiento valorados
    con los de test
    '''
    clf = linear_model.SGDClassifier(max_iter=10000, tol=1e-6)
    clf.fit(train_data, train_labels)
    return clf.score(test_data, test_labels)

def pruebaLogisticRegression(train_data, test_data, train_labels, test_labels):
    '''
    @brief Función que aplica regresión logística y obtiene la valoración del ajuste.
    @param train_data Datos de entrenamiento
    @param test_data Datos de test
    @param train_labels Etiquetas de los datos de entrenamiento
    @param test_labels Etiquetas de los datos de test
    @return Devuelve el score del ajuste con los datos de entrenamiento valorados
    con los de test
    '''
    clf = linear_model.LogisticRegression(max_iter=10000, random_state=0, solver='lbfgs', multi_class='multinomial')
    clf.fit(train_data, train_labels)
    return clf.score(test_data, test_labels)

def pruebaPassiveAgressive(train_data, test_data, train_labels, test_labels):
    '''
    @brief Función que aplica modelos pasivo-agresivos y obtiene la valoración del ajuste.
    @param train_data Datos de entrenamiento
    @param test_data Datos de test
    @param train_labels Etiquetas de los datos de entrenamiento
    @param test_labels Etiquetas de los datos de test
    @return Devuelve el score del ajuste con los datos de entrenamiento valorados
    con los de test
    '''
    clf = linear_model.PassiveAggressiveClassifier(max_iter=1000, random_state=0, tol=1e-3)
    clf.fit(train_data, train_labels)
    return clf.score(test_data, test_labels)

def pruebaPerceptron(train_data, test_data, train_labels, test_labels):
    '''
    @brief Función que aplica perceptrón y obtiene la valoración del ajuste.
    @param train_data Datos de entrenamiento
    @param test_data Datos de test
    @param train_labels Etiquetas de los datos de entrenamiento
    @param test_labels Etiquetas de los datos de test
    @return Devuelve el score del ajuste con los datos de entrenamiento valorados
    con los de test
    '''
    clf = linear_model.Perceptron(tol=1e-3, random_state=0)
    clf.fit(train_data, train_labels)
    return clf.score(test_data, test_labels)

def pruebaAlgoritmos():
    algoritmos = [pruebaMinimosCuadradosRL, pruebaRidge, pruebaLasso, pruebaLARSLasso, pruebaSGDClassifier, pruebaLogisticRegression, pruebaPassiveAgressive, pruebaPerceptron]
    nombre_algoritmos = ["Mínimos cuadrados", "Ridge", "Lasso", "LARS Lasso", "SGD", "Logistic Regression", "Passive-Agressive", "Perceptron"]
    dataset, labels=readData()
    train_data, test_data, train_labels, test_labels = train_test_split(dataset, labels, stratify=labels, train_size=0.2, test_size=0.8)
    for algoritmo,nombre in zip(algoritmos,nombre_algoritmos):
        score=algoritmo(train_data, test_data, train_labels, test_labels)
        print("El score obtenido por el algoritmo " + nombre + " es: " + str(score))

pruebaAlgoritmos()
