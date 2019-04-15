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
    # Obtenemos los datos, las etiquetas, los nombres de las características y los
    # nombres de las clases
    data = iris["data"]
    labels = iris["target"]
    nombre_clases = iris["target_names"]
    nombre_features = iris["feature_names"]
    # Nos quedamos con las dos últimas columnas de los datos
    last2_col = data[:,-2:]
    # Dividimos los datos según las etiquetas en una lista de listas
    # Cada sublista tiene el conjunto de datos que corresponde a cada clase, esto
    # es, en la posición 0 la lista de elementos con clase 0, en la posición 1 la
    # lista de elementos con la clase 1, etc
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
    # Obtenemos los datosy las etiquetas
    data = iris["data"]
    labels = iris["target"]
    # Calculamos el tamaño del test y train
    test_size = int(test_percentage*len(data))
    train_size = len(data)-test_size

    #Dividimos los datos por clases, esto es una lista con sublistas que contienen en cada
    # posición los elementos de la clase correspondiente, es decir [[elementos_clase0], [elementos_clase1], [elementos_clase2]]
    max_label = max(labels)
    data_divided = []
    for i in range(max_label+1):
        subsample = []
        for j in range(len(data)):
            if labels[j]==i:
                # Lo pongo como una lista para que se vea mejor cuando lo imprima
                subsample.append(list(data[j]))
        # Antes de añadir la sublista la barajamos para obtener una muestra aleatoria
        np.random.shuffle(subsample)
        data_divided.append(subsample)
    # Obtenemos el porcentaje de aparición de cada clase en el conjunto de datos original
    classes_percentage = [len(data_divided[i])/len(data) for i in range(len(data_divided))]
    # Obtenemos el número de elementos de cada clase que corresponden para el test
    elem_per_class_test = [int(test_size*classes_percentage[i]) for i in range(len(classes_percentage))]

    # Obtenemos los conjuntos de datos de train y test
    test_data = []
    train_data = []
    # Nos quedamos para el test con los primeros elementos (en este caso los 10 primeros)
    # aprovechandonos del hecho de que hemos barajado previamente cada una de las sublistas
    for i in range(len(data_divided)):
        test_data = test_data+data_divided[i][:elem_per_class_test[i]]
        train_data = train_data+data_divided[i][elem_per_class_test[i]:]
    # Obtenemos las etiquetas de train y test
    test_labels = []
    train_labels = []
    # Generamos una lista que contiene las etiquetas de test y train, en este caso al haber
    # 10 instancias de cada clase en el conjunto de test este bucle lo que hace es concatenar los
    # vectores [0,...,0]+[1,...,1]+[2,...,2] siendo los 3 de longitud 10.
    for i in range(len(elem_per_class_test)):
        test_labels = test_labels+(elem_per_class_test[i]*[i])
        train_labels = train_labels+(len(data_divided[i])-elem_per_class_test[i])*[i]

    # Imprimimos los conjuntos de datos de test y train
    print("Datos de test")
    print(test_data)
    print("\n\n\nDatos de train")
    print(train_data)

    # Contamos los elementos que hay en cada clase para el conjunto de test y train
    # y lo mostramos por pantalla
    unique_test, counts_test = np.unique(test_labels, return_counts=True)
    print("Distribución de clases en test")
    print(dict(zip(unique_test, counts_test)))

    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    print("Distribución de clases en train")
    print(dict(zip(unique_train, counts_train)))
    print("\n\n")

    # Calculamos los porcentajes obtenidos en el train y en el test
    # Sabemos que la proporción de datos es de 50,50,50 para las clases, esto es un 33% de aparición
    # de cada clase
    # Los porcentajes salen exactamente un 33% porque el cálculo es exacto
    print("Porcentaje de clases en test")
    print(dict(zip(unique_test, counts_test/np.sum(counts_test))))
    print("Porcentaje de clases en train")
    print(dict(zip(unique_train, counts_train/np.sum(counts_train))))

    input("Pulsa ENTER para seguir")


################################################################################
##                                   Parte 3                                  ##
################################################################################

def parte3(nvalues=100):
    # Obtenemos nvalues valores equiespaciados entre 0 y 2pi
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
