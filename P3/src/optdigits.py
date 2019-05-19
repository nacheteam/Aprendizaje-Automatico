# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import decomposition

from sklearn.manifold import TSNE

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
##                       REDUCCIÓN DE DIMENSIONALIDAD                         ##
################################################################################

def nComponentsCriterion(vratio, explained_var=0.95):
    '''
    @brief Función que da el número de componentes que explican al menos un 95%
    de la varianza dado el resultado de un PCA o FA.
    @param vratio Porcentaje de varianza explicada para cada componente
    @param explained_var Mínimo que requerimos de varianza explicada
    @return Devuelve el número de componentes que explican al menos un
    95% de la varianza
    '''
    index = 0
    sum_var = 0
    for i in range(len(vratio)):
        sum_var+=vratio[i]
        if sum_var>=explained_var:
            index=i
            break
    return index

def reducePCA(dataset):
    '''
    @brief Función que aplica una reducción PCA al conjunto de datos
    @param dataset Conjunto de datos
    @return Devuelve el conjunto de datos reducido usando como número de componentes
    aquel que explique al menos un 95% de la varianza
    '''
    pca = decomposition.PCA().fit(dataset)
    n_comp = nComponentsCriterion(pca.explained_variance_ratio_)
    pca = decomposition.PCA(n_components=n_comp).fit(dataset)
    return pca.transform(dataset)

def reduceIncrementalPCA(dataset):
    '''
    @brief Función que aplica una reducción Incremental PCA al conjunto de datos
    @param dataset Conjunto de datos
    @return Devuelve el conjunto de datos reducido usando como número de componentes
    aquel que explique al menos un 95% de la varianza
    '''
    ipca = decomposition.IncrementalPCA().fit(dataset)
    n_comp = nComponentsCriterion(ipca.explained_variance_ratio_)
    ipca = decomposition.IncrementalPCA(n_components=n_comp).fit(dataset)
    return ipca.transform(dataset)

def reduceKernelPCA(dataset):
    '''
    @brief Función que aplica una reducción Kernel PCA al conjunto de datos
    @param dataset Conjunto de datos
    @return Devuelve el conjunto de datos reducido usando como número de componentes
    aquel que explique al menos un 95% de la varianza
    '''
    kpca = decomposition.KernelPCA().fit(dataset)
    kpca_transform = kpca.fit_transform(dataset)
    explained_variance = np.var(kpca_transform, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    n_comp = nComponentsCriterion(explained_variance_ratio)
    kpca = decomposition.KernelPCA(n_components=n_comp).fit(dataset)
    return kpca.transform(dataset)

def reduceFactorAnalysis(dataset):
    '''
    @brief Función que aplica una reducción usando Factor Analysis al conjunto de datos
    @param dataset Conjunto de datos
    @return Devuelve el conjunto de datos reducido usando como número de componentes
    aquel que explique al menos un 95% de la varianza
    '''
    fa = decomposition.FactorAnalysis().fit(dataset)
    fa_transform = fa.fit_transform(dataset)
    explained_variance = np.var(fa_transform, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    n_comp = nComponentsCriterion(explained_variance_ratio)
    fa = decomposition.FactorAnalysis(n_components=n_comp).fit(dataset)
    return fa.transform(dataset)

def pruebaReduccion(dataset):
    '''
    @brief Función que devuelve todos los conjuntos de datos que surgen al aplicar
    reducciones PCA, Incremental PCA, Kernel PCA y Factor Analisys.
    @param dataset Conjunto de datos
    @return Devuelve una lista de nombres de los algoritmos empleados en la reducción
    y una lista con los conjuntos de datos resultantes.
    '''
    nombres = ["PCA", "Incremental PCA", "Kernel PCA", "Factor Analysis"]
    datasets = [reducePCA(dataset), reduceIncrementalPCA(dataset), reduceKernelPCA(dataset), reduceFactorAnalysis(dataset)]
    return nombres, datasets

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
    return preprocessing.scale(dataset)

def normalizeData(dataset):
    '''
    @brief Función que realiza una normalización de los datos para que tengan
    norma 1 según la norma L2 o euclídea
    @param dataset Conjunto de datos
    @return Devuelve el conjunto de datos normalizado
    '''
    return preprocessing.normalize(dataset,norm="l2")

def pruebaPreprocesamiento(dataset):
    '''
    @brief Función que devuelve todas las combinaciones interesantes de conjuntos
    preprocesados para aplicar los modelos
    @param dataset Conjunto de datos
    @return Devuelve dos listas, una con las cadenas correspondientes al preprocesado
    aplicado y otra con la lista de conjuntos tras aplicar el preprocesado
    '''
    nombres = ["Estandarización", "Normalización", "Normalización y después estandarización"]
    datasets = [scaleData(dataset), normalizeData(dataset), scaleData(normalizeData(dataset))]
    return nombres, datasets

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

def pruebaAlgoritmos(dataset,labels,algoritmos = [pruebaMinimosCuadradosRL, pruebaRidge, pruebaLasso, pruebaSGDClassifier, pruebaLogisticRegression, pruebaPassiveAgressive, pruebaPerceptron], nombre_algoritmos = ["Mínimos cuadrados", "Ridge", "Lasso", "SGD", "Logistic Regression", "Passive-Agressive", "Perceptron"]):
    train_data, test_data, train_labels, test_labels = train_test_split(dataset, labels, stratify=labels, train_size=0.2, test_size=0.8)
    scores = []
    for algoritmo,nombre in zip(algoritmos,nombre_algoritmos):
        score=algoritmo(train_data, test_data, train_labels, test_labels)
        scores.append(score)
        print("El score obtenido por el algoritmo " + nombre + " es: " + str(score))
    return scores

################################################################################
##                       AJUSTE DEL ALGORITMO RL                              ##
################################################################################

def regresionLogistica(data,labels):
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, stratify=labels, train_size=0.2, test_size=0.8)
    clf = linear_model.LogisticRegression(max_iter=10000, tol=1e-5, C=1.0, random_state=123456789, solver='lbfgs', multi_class='multinomial', verbose=1)
    clf.fit(train_data, train_labels)
    return clf.score(test_data, test_labels)

################################################################################
##                     FUNCIONES DE VISUALIZACIÓN                             ##
################################################################################

def visualizacionTSNE(data,labels):
    data_red = TSNE(n_components=2).fit_transform(data)
    data_divided = [[],[],[],[],[],[],[],[],[],[]]
    colores = []
    for d,l in zip(data_red,labels):
        data_divided[l].append(list(d))
    clase = 0
    for i in range(len(data_divided)):
        data_divided[i]=np.array(data_divided[i])
    for data,col in zip(data_divided,COLORES[:10]):
        plt.scatter(data[:,0], data[:,1], c=col, label="Clase " + str(clase))
        clase+=1
    plt.legend()
    plt.title("Conjunto optdigits")
    plt.show()

################################################################################
##                                MAIN                                        ##
################################################################################

data,labels = readData()

s = regresionLogistica(scaleData(data),labels)
print("Score: " + str(s))

'''
visualizacionTSNE(data,labels)
input("PULSE ENTER PARA CONTINUAR")
'''

'''
# Primero probamos todos los algoritmos sin preprocesamiento ni reducción de dimensionalidad
print("TODOS LOS ALGORITMOS")
print("########################################################################")
print("Sin reducción de dimensionalidad")
print("Sin preprocesamiento")
print("########################################################################")
pruebaAlgoritmos(data,labels)
print("\n\n\n\n")

# Probamos sólo los algoritmos que funcionan bien con todas las posibilidades
algoritmos = [pruebaSGDClassifier, pruebaLogisticRegression, pruebaPassiveAgressive, pruebaPerceptron]
nombre_algoritmos = ["SGD", "Logistic Regression", "Passive-Agressive", "Perceptron"]
print("SÓLO LOS ALGORITMOS QUE FUNCIONAN BIEN")
print("Algoritmos: " + str(nombre_algoritmos))

print("\n########################################################################")
print("Sin reducción de dimensionalidad")
print("Sin preprocesamiento")
print("########################################################################")
pruebaAlgoritmos(data,labels,algoritmos,nombre_algoritmos)

nombres_preprocesamiento, datasets_preprocesados = pruebaPreprocesamiento(data)
for nom_pre,dataset_pre in zip(nombres_preprocesamiento,datasets_preprocesados):
    print("\n########################################################################")
    print("Sin reducción de dimensionalidad")
    print("Aplicado el preprocesamiento: " + nom_pre)
    print("########################################################################")
    pruebaAlgoritmos(dataset_pre,labels,algoritmos,nombre_algoritmos)
print("------------------------------------------------------------------------\n\n")

nombres_reduccion, reduced_datasets = pruebaReduccion(data)
for nom_red,dataset_red in zip(nombres_reduccion,reduced_datasets):
    nombres_preprocesamiento, datasets_preprocesados = pruebaPreprocesamiento(dataset_red)
    print("\n########################################################################")
    print("Con reducción de dimensionalidad de tipo: " + nom_red + " con número de variables: " + str(len(dataset_red[0])))
    print("Sin preprocesamiento")
    print("########################################################################")
    pruebaAlgoritmos(dataset_red,labels,algoritmos,nombre_algoritmos)
    for nom_pre,dataset_pre in zip(nombres_preprocesamiento,datasets_preprocesados):
        print("\n########################################################################")
        print("Con reducción de dimensionalidad de tipo: " + nom_red + " con número de variables: " + str(len(dataset_red[0])))
        print("Aplicado el preprocesamiento: " + nom_pre)
        print("########################################################################")
        pruebaAlgoritmos(dataset_pre,labels,algoritmos,nombre_algoritmos)
    print("------------------------------------------------------------------------\n\n")
'''
