# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

# Inicializamos la semilla
np.random.seed(123456789)

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

def pruebaMinimosCuadradosRL(dataset, labels):
    reg = linear_model.LinearRegression().fit(dataset,labels)
    return reg.score(dataset,labels)

def pruebaRidge(dataset, labels):
    clf=linear_model.Ridge(alpha=0.1)
    clf.fit(dataset, labels)
    return clf.score(dataset, labels)

def pruebaLasso(dataset, labels):
    reg = linear_model.Lasso(alpha=0.1)
    reg.fit(dataset, labels)
    return reg.score(dataset, labels)

def pruebaLARSLasso(dataset, labels):
    reg = linear_model.LassoLars(alpha=0.1)
    reg.fit(dataset, labels)
    return reg.score(dataset, labels)

# Este da un 96% como los champions
def pruebaSGDClassifier(dataset, labels):
    clf = linear_model.SGDClassifier(max_iter=10000, tol=1e-6)
    clf.fit(dataset, labels)
    return clf.score(dataset, labels)

# Este un 99.71%
def pruebaLogisticRegression(dataset, labels):
    clf = linear_model.LogisticRegression(max_iter=10000, random_state=0, solver='lbfgs', multi_class='multinomial')
    clf.fit(dataset, labels)
    return clf.score(dataset, labels)


dataset, labels=readData()
score = pruebaLogisticRegression(dataset, labels)
print(score)
