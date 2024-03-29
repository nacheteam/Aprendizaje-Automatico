# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import metrics
from sklearn.manifold import TSNE

import pandas as pd
import seaborn as sns

# Inicializamos la semilla
np.random.seed(123456789)

################################################################################
##                                COLORES RGB                                 ##
################################################################################

AMARILLO = (1,1,0.2)
ROJO = (1,0,0)
NARANJA = (1,0.5,0)
VERDE = (0.5,1,0)
VERDE_AZULADO = (0,1,0.5)
AZUL_CLARO = (0,1,1)
AZUL = (0,0,1)
MORADO = (0.5,0,1)
ROSA = (1,0,1)
GRIS = (0.5,0.5,0.5)
NEGRO = (0,0,0)

# Se usan para colorear los puntos según la clase
COLORES = [AMARILLO,ROJO,NARANJA,VERDE,VERDE_AZULADO,AZUL_CLARO,AZUL,MORADO,ROSA,GRIS,NEGRO]
COLORES_LABEL = ["Amarillo","Rojo","Naranja","Verde","Verde azulado","Azul claro","Azul","Morado","Rosa","Gris","Negro"]

################################################################################
##                           FUNCIONES AUXILIARES                             ##
################################################################################

def readData(path="./datos/"):
    '''
    @brief Función que lee los ficheros de airfoil y los unifica en una matriz de
    datos y un vector de labels
    @param path Carpeta en la que están los datos
    '''
    # Abrimos el fichero de datos
    file = open(path+"/airfoil_self_noise.dat", "r")

    data = []
    output = []
    for line in file:
        d = []
        for el in line.split("\t")[:-1]:
            d.append(float(el))
        data.append(d)
        output.append(float(line.split("\t")[-1]))
    return np.array(data), np.array(output)

################################################################################
##                            PREPROCESAMIENTO                                ##
################################################################################

def scaleData(dataset):
    '''
    @brief Función que realiza una estandarización de los conjuntos de test y train
    a media cero y escalados según la varianza.
    @param dataset Conjunto de datos
    @return Devuelve dos conjuntos de datos estandarizados
    '''
    return preprocessing.StandardScaler().fit(dataset).transform(dataset)

def normalizeData(dataset):
    '''
    @brief Función que realiza una normalización de los datos para que tengan
    norma 1 según la norma L2 o euclídea
    @param dataset Conjunto de datos
    @return Devuelve el conjunto de datos normalizado
    '''
    return preprocessing.normalize(dataset,norm="l2")

def polyData(dataset):
    '''
    @brief Función que añade nuevos valores de orden polinómico al conjunto de datos
    @param dataset Conjunto de datos
    @return Devuelve el conjunto de datos con datos añadidos
    '''
    return preprocessing.PolynomialFeatures(degree=3).fit_transform(dataset)

def raizDatos(dataset):
    '''
    @brief Función que aplica la raíz cuadrada a los datos
    @return Devuelve el conjunto de datos transformados
    '''
    transformer = preprocessing.FunctionTransformer(np.sqrt)
    return transformer.transform(dataset)

def logDatos(dataset):
    '''
    @brief Función que aplica una transformación logarítmica log(1+x) a los datos
    @return Devuelve el conjunto de datos transformados
    '''
    transformer = preprocessing.FunctionTransformer(np.log1p)
    return transformer.transform(dataset)


def pruebaPreprocesamiento(dataset):
    '''
    @brief Función que devuelve todas las combinaciones interesantes de conjuntos
    preprocesados para aplicar los modelos
    @param dataset Conjunto de datos
    @return Devuelve dos listas, una con las cadenas correspondientes al preprocesado
    aplicado y otra con la lista de conjuntos tras aplicar el preprocesado
    '''
    nombres = ["Sin preprocesamiento", "Raíz cuadrada", "Logaritmo", "Logaritmo+estandarizado","Raíz cuadrada + estandarizado", "Datos polinomicos+raiz cuadrada+estandarizado", "Datos polinomicos+logaritmo+estandarizado","Estandarización", "Normalización", "Datos polinómicos", "Datos polinómicos estandarizados", "Datos polinómicos normalizados", "Datos polinómicos estandarizados y normalizados","Normalización y después estandarización"]
    datasets = [dataset, raizDatos(dataset), logDatos(dataset), scaleData(logDatos(dataset)),scaleData(raizDatos(dataset)), scaleData(polyData(raizDatos(dataset))), scaleData(polyData(logDatos(dataset))),scaleData(dataset), normalizeData(dataset), polyData(dataset), scaleData(polyData(dataset)), normalizeData(polyData(dataset)), scaleData(normalizeData(dataset)), scaleData(normalizeData(polyData(dataset)))]
    return nombres, datasets

################################################################################
##                 FUNCIONES DE PRUEBA DE ALGORITMOS                          ##
################################################################################

def pruebaMinimosCuadradosRL(train_data, test_data, train_out, test_out):
    '''
    @brief Función que aplica mínimos cuadrados y obtiene la valoración del ajuste.
    @param train_data Datos de entrenamiento
    @param test_data Datos de test
    @param train_out Output de los datos de entrenamiento
    @param test_out Output de los datos de test
    @return Devuelve el score del ajuste con los datos de entrenamiento valorados
    con los de test
    '''
    reg = linear_model.LinearRegression().fit(train_data,train_out)
    return reg.score(test_data,test_out)

def pruebaRidge(train_data, test_data, train_out, test_out):
    '''
    @brief Función que aplica el modelo de Ridge y obtiene la valoración del ajuste.
    @param train_data Datos de entrenamiento
    @param test_data Datos de test
    @param train_out Output de los datos de entrenamiento
    @param test_out Output de los datos de test
    @return Devuelve el score del ajuste con los datos de entrenamiento valorados
    con los de test
    '''
    clf=linear_model.Ridge(alpha=0.1, max_iter=10000000)
    clf.fit(train_data, train_out)
    return clf.score(test_data, test_out)

def pruebaLasso(train_data, test_data, train_out, test_out):
    '''
    @brief Función que aplica LASSO y obtiene la valoración del ajuste.
    @param train_data Datos de entrenamiento
    @param test_data Datos de test
    @param train_out Output de los datos de entrenamiento
    @param test_out Output de los datos de test
    @return Devuelve el score del ajuste con los datos de entrenamiento valorados
    con los de test
    '''
    reg = linear_model.Lasso(alpha=0.2, max_iter=1000000)
    reg.fit(train_data, train_out)
    return reg.score(test_data, test_out)

def pruebaElasticNet(train_data, test_data, train_out, test_out):
    '''
    @brief Función que aplica ElasticNet y obtiene la valoración del ajuste
    @param train_data Datos de entrenamiento
    @param test_data Datos de test
    @param train_out Output de los datos de entrenamiento
    @param test_out Output de los datos de test
    @return Devuelve el score del ajuste con los datos de entrenamiento valorados
    con los de test
    '''
    clf = linear_model.ElasticNet(random_state=0, max_iter=1000000)
    clf.fit(train_data, train_out)
    return clf.score(test_data, test_out)

def pruebaLars(train_data, test_data, train_out, test_out):
    '''
    @brief Función que aplica Lars y obtiene la valoración del ajuste.
    @param train_data Datos de entrenamiento
    @param test_data Datos de test
    @param train_out Output de los datos de entrenamiento
    @param test_out Output de los datos de test
    @return Devuelve el score del ajuste con los datos de entrenamiento valorados
    con los de test
    '''
    clf = linear_model.Lars(n_nonzero_coefs=1)
    clf.fit(train_data, train_out)
    return clf.score(test_data, test_out)

def pruebaLassoLars(train_data, test_data, train_out, test_out):
    '''
    @brief Función que aplica LASSO-LARS y obtiene la valoración del ajuste.
    @param train_data Datos de entrenamiento
    @param test_data Datos de test
    @param train_out Output de los datos de entrenamiento
    @param test_out Output de los datos de test
    @return Devuelve el score del ajuste con los datos de entrenamiento valorados
    con los de test
    '''
    clf = linear_model.LassoLars(alpha=0.01, max_iter=1000000)
    clf.fit(train_data, train_out)
    return clf.score(test_data, test_out)

def pruebaBayesianRidge(train_data, test_data, train_out, test_out):
    '''
    @brief Función que aplica Bayesian Ridge y obtiene la valoración del ajuste.
    @param train_data Datos de entrenamiento
    @param test_data Datos de test
    @param train_out Output de los datos de entrenamiento
    @param test_out Output de los datos de test
    @return Devuelve el score del ajuste con los datos de entrenamiento valorados
    con los de test
    '''
    clf = linear_model.BayesianRidge()
    clf.fit(train_data, train_out)
    return clf.score(test_data, test_out)

def pruebaAlgoritmos(data, out,algoritmos = [pruebaMinimosCuadradosRL, pruebaRidge, pruebaLasso, pruebaElasticNet, pruebaLars, pruebaLassoLars, pruebaBayesianRidge], nombre_algoritmos = ["Mínimos cuadrados", "Ridge", "Lasso", "ElasticNet", "Lars", "Lasso-Lars", "Bayesian Ridge"]):
    '''
    @brief Función que se encarga de probar todos los algoritmos pasados por la lista
    algoritmos dividiendo el conjunto de datos en train y test y pasando estos argumentos
    a las funciones que implementan cada algoritmo.
    @param dataset Conjunto de datos
    @param out Salida de los datos
    @param algoritmos Nombres de las funciones que implementan los algoritmos
    @param nombre_algoritmos Cadenas de texto que contienen los nombres de los algoritmos.
    @return Devuelve una lista de scores de los algoritmos.
    '''
    train_data, test_data, train_out, test_out = train_test_split(data, out, train_size=0.2, test_size=0.8)
    scores = []
    for algoritmo,nombre in zip(algoritmos,nombre_algoritmos):
        score=algoritmo(train_data, test_data, train_out, test_out)
        scores.append(score)
        print("El score obtenido por el algoritmo " + nombre + " es: " + str(score))
    return scores

################################################################################
##                            AJUSTA EL MODELO                                ##
################################################################################

def minimosCuadradosRL(train_data, test_data, train_out, test_out):
    '''
    @brief Función que aplica mínimos cuadrados y obtiene la valoración del ajuste.
    @param train_data Datos de entrenamiento
    @param test_data Datos de test
    @param train_out Output de los datos de entrenamiento
    @param test_out Output de los datos de test
    @return Devuelve el score del ajuste con los datos de entrenamiento valorados
    con los de test
    '''
    reg = linear_model.LinearRegression().fit(train_data,train_out)
    return reg

################################################################################
##                     FUNCIONES DE VISUALIZACIÓN                             ##
################################################################################

def pairPlotDatos(dataset, title):
    '''
    @brief Función que realiza un pairplot de los datos para visualización
    @param dataset Conjunto de datos
    @param title Título del plot
    '''
    data_frame = pd.DataFrame(data=dataset, columns=["Frequency", "Angle-attack", "Chord-length", "Free-stream vel", "Suc. displacement thick."])
    g = sns.pairplot(data_frame, kind="reg", palette="husl",plot_kws={'line_kws':{'color':'red'}})
    g.fig.suptitle(title, size=16)
    g.fig.subplots_adjust(top=.93)
    g.fig.show()

################################################################################
##                                MAIN                                        ##
################################################################################

data,out = readData()

# Visualización
datasets = [data, raizDatos(data), scaleData(data), normalizeData(data), logDatos(data)]
nombres = ["Sin preprocesado", "Raíz cuadrada", "Estandarizados", "Normalizados", "Logaritmo(1+x)"]
for n,d in zip(nombres,datasets):
    if not "polinomicos" in n and not "polinómicos" in n and not "polinomico" in n and not "polinómico" in n:
        print("Pairplot con el preprocesamiento: " + n)
        pairPlotDatos(d,n)
        input("PRESIONE ENTER PARA CONTINUAR")

# Con todos los algoritmos
print("TODOS LOS ALGORITMOS")
nombres, datasets = pruebaPreprocesamiento(data)
for dataset,nombre in zip(datasets, nombres):
    print("#####################################################################")
    print("Preprocesamiento: " + nombre)
    print("#####################################################################")
    pruebaAlgoritmos(dataset,out)
    print("\n\n")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n\n\n\n")

# Eliminamos Lars
nombres, datasets = pruebaPreprocesamiento(data)
algoritmos = [pruebaMinimosCuadradosRL, pruebaRidge, pruebaLasso, pruebaElasticNet, pruebaLassoLars, pruebaBayesianRidge]
nombre_algoritmos = ["Mínimos cuadrados", "Ridge", "Lasso", "ElasticNet", "Lasso-Lars", "Bayesian Ridge"]
print("SOLO LOS ALGORITMOS QUE FUNCIONAN BIEN")
print("Algoritmos: " + str(nombre_algoritmos))
for dataset,nombre in zip(datasets, nombres):
    print("#####################################################################")
    print("Preprocesamiento: " + nombre)
    print("#####################################################################")
    pruebaAlgoritmos(dataset,out,algoritmos,nombre_algoritmos)
    print("\n\n")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n\n\n\n")


# Computamos los scores
train_data, test_data, train_out, test_out = train_test_split(scaleData(polyData(raizDatos(data))), out, train_size=0.2, test_size=0.8)
clf = minimosCuadradosRL(train_data, test_data, train_out, test_out)
print("Score: " + str(clf.score(test_data,test_out)))
out_pred = clf.predict(test_data)
print("Mean Squared Error: " + str(metrics.mean_squared_error(test_out,out_pred)))
print("Explained variance score: "  + str(np.mean(metrics.explained_variance_score(test_out,out_pred))))
