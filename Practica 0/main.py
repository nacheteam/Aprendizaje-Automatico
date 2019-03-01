import numpy as np
import sklearn as sk
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

# Se usan para colorear los puntos según la clase
COLORES = [AMARILLO,ROJO,NARANJA,VERDE,VERDE_AZULADO,AZUL_CLARO,AZUL,MORADO,ROSA,GRIS]
COLORES_LABEL = ["Amarillo","Rojo","Naranja","Verde","Verde azulado","Azul claro","Azul","Morado","Rosa","Gris"]

################################################################################
##                                   Parte 1                                  ##
################################################################################

def parte1():
    # Cargamos el dataset
    iris = datasets.load_iris()
    # Obtenemos los datos, las etiquetas y los nombres de las características
    data = iris["data"]
    labels = iris["target"]
    nombre_clases = iris["feature_names"]
    # Nos quedamos con las dos últimas columnas de los datos
    last2_col = data[:,-2:]
    # Dividimos las filas según las etiquetas
    max_label = max(labels)
    data_divided = []
    for i in range(max_label+1):
        subsample = []
        for j in range(len(data)):
            if labels[j]==i:
                subsample.append(last2_col[j])
        data_divided.append(np.array(subsample))
    # Pintamos los datos por colores en función de las clases
    for i in range(len(data_divided)):
        subsample = data_divided[i]
        plt.scatter(subsample[:,:1],subsample[:,1:],c=COLORES[i],label=COLORES_LABEL[i]+" ("+nombre_clases[i]+")")
    plt.xlabel(nombre_clases[-2])
    plt.ylabel(nombre_clases[-1])
    plt.title("Scatter Plot de las dos últimas características")
    plt.legend()
    plt.show()

################################################################################
##                                   Parte 2                                  ##
################################################################################

def parte2():
    # Cargamos el dataset
    iris = datasets.load_iris()
    # Obtenemos los datos, las etiquetas y los nombres de las características
    data = iris["data"]
    labels = iris["target"]
    # Separamos los datos en test y train
    train_data,test_data,train_labels,test_labels = train_test_split(data,labels,test_size=0.20,stratify=labels)
    # Imprimimos los datos
    print("Datos de test")
    print(test_data)
    print(test_labels)
    print("Datos de entrenamiento")
    print(train_data)
    print(train_labels)
    input("Pulsa ENTER para seguir")

################################################################################
##                                 MAIN                                       ##
################################################################################

def main():
    parte1()
    parte2()

main()
