import numpy as np
import sklearn as sk
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(123456)

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
##                                   Parte 1                                  ##
################################################################################

def parte1():
    # Cargamos el dataset
    iris = datasets.load_iris()
    # Obtenemos los datos, las etiquetas y los nombres de las características
    data = iris["data"]
    labels = iris["target"]
    nombre_clases = iris["target_names"]
    nombre_features = iris["feature_names"]
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
    plt.xlabel(nombre_features[-2])
    plt.ylabel(nombre_features[-1])
    plt.title("Scatter Plot de las dos últimas características")
    plt.legend()
    plt.show()

################################################################################
##                                   Parte 2                                  ##
################################################################################

def parte2():
    # Porcentaje de test y train
    test_percentage = 0.2
    train_percentage = 0.8
    # Cargamos el dataset
    iris = datasets.load_iris()
    # Obtenemos los datos, las etiquetas y los nombres de las características
    data = iris["data"]
    labels = iris["target"]
    # Calculamos el tamaño del test y train
    test_size = int(test_percentage*len(data))
    train_size = len(data)-test_size
    # Calculamos números de forma aleatoria (según una uniforme) de para sacar los índices
    # del conjunto de test
    test_index = np.random.choice(len(data), test_size,replace=False)
    train_index = np.array(list(set([i for i in range(len(data))]).difference(set(test_index))))
    # Obtenemos los conjuntos de train y test
    test_data = data[test_index]
    train_data = data[train_index]
    test_labels = labels[test_index]
    train_labels = labels[train_index]

    unique_test, counts_test = np.unique(test_labels, return_counts=True)
    print("Distribución de clases en test")
    print(dict(zip(unique_test, counts_test)))

    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    print("Distribución de clases en train")
    print(dict(zip(unique_train, counts_train)))

    input("Pulsa ENTER para seguir")

################################################################################
##                                   Parte 3                                  ##
################################################################################

def parte3(nvalues=100):
    # Fijamos la semilla
    np.random.seed(12345)
    # Obtenemos nvalues valores equiespaciados entre 0  y 2pi
    values = np.linspace(0, 2*np.pi, num=nvalues)
    # Calculamos el seno, coseno y seno+coseno
    sine = np.sin(values)
    cosine = np.cos(values)
    sine_plus_cosine = sine+cosine
    # Dibujamos las curvas resultantes
    plt.plot(values,sine,"--",c=NEGRO,label="Seno")
    plt.plot(values,cosine,"--",c=AZUL,label="Coseno")
    plt.plot(values,sine_plus_cosine,"--",c=ROJO,label="Seno más coseno")
    plt.legend()
    plt.show()

################################################################################
##                                 MAIN                                       ##
################################################################################

def main():
    parte1()
    parte2()
    parte3()

main()
