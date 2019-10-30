# -*- coding: utf-8 -*-
"""
EXAMEN PYTHON: El proposito de este examen es comprobar si el candidato posee conocimientos de Python (básico y aplicados a ML)
Intente usar Python 3 y puede usar las siguientes librerias (salvo que se indique lo contrario): numpy, pandas, matplotlib.pyplot y sklearn
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

"""
BLOQUE 1 CONOCIMIENTOS GENERALES DE PYTHONb

Pregunta 1.1: crea una funcion a la cual le pases por parametros el 
numero de filas y el numero de columnas y te la rellene con numeros aleatorios del 1 al 100 con 2 decimales
Crea una matriz de 10 filas y 7 columnas. (No utilices NUMPY ni PANDAS en este apartado, en los demás puedes usarlo)
"""

"Usando numpy"
def aleatoria(x,y):
    "Matriz xy con número aleatorios de un rango definido, en este caso de 0 a 100"
    matriz=np.random.randint(1,101,(x,y))/100
    return matriz

matriz = aleatoria(10,7)

"Sin usar numpy"

from random import uniform

def alenonumpy(x,y):
    matrizno = [[round(uniform(1,101),2) for a in range(y)] for b in range(x)]
    return matrizno

matrizno= alenonumpy(10,7)

print(matriz)
print(matrizno)

"""
Pregunta 1.2: A partir de la matriz anterior crea un dataframe de PANDAS, poniendo como nombre de columnas los días de la semana
Si no pudiste crear la matriz del apartado anterior, puedes crear una a partir de NUMPY
"""
matriz_semana = pd.DataFrame(matriz, columns = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"])

print(matriz_semana)

"""
pregunta 1.3: Como crearías una estructura de datos si el tipo de dato no fuera uniforme (mismo tipo de dato)?
Crea una estructura cuyo primer elemento sea un vector de NUMPY de 10 números aleatorios entre 1 y 100
Y el segundo elemento un vector con esos mismos valores pero como caracteres.
"""
a = np.random.randint(1,101,10)

lista = [a,str(a)]


"""
Pregunta 1.4: normaliza todos los elementos de la primera matriz. Esto es, por
cada elemento resta la media y divide por la desviacion tipica del conjunto.
"""
n_matriz = (matriz - np.mean(matriz))/np.std(matriz)

"""
Pregunta 1.5: Filtra la matriz normalizada y devuelve los valores entre -1 y 1, sin usar un bucle
El resultado debe ser un vector con dichos valores ordenados descendentemente
"""
redu_matriz = sorted(np.extract(np.extract(n_matriz>-1,n_matriz) <1,np.extract(n_matriz>-1,n_matriz)),reverse=True)

print(redu_matriz)
"""
BLOQUE 2: CREAR HISTOGRAMA Y DIAGRAMA DE BARRAS DE MÁS FRECUENTES.

2.1 Crea un vector de números enteros aleatorios entre el 1 y el 10, de tamaño 100
Crea otro vector de numeros aleatorios entre el 11 y el 20 del mismo tamaño
y muestra por pantalla la tabla de frecuencia de la unión de los 2 vectores
para los valores 5,6 y 8 (de la primera) y para los valores 14,16,19 (de la segunda).
"""
v1 = np.random.randint(1,11,100)
v2 = np.random.randint(11,21,100)
v3 = np.append(v1,v2)

from collections import Counter
cuenta = Counter(v3)
elegidos =[5,6,8,14,16,19]
resumen = {k:v for k,v in cuenta.items() if k in elegidos}

plt.bar(x=resumen.keys(),height = resumen.values())
plt.show()



"""
2.2 crea el histograma para dicho vector de frecuencias, con ancho de barra 0.75, título: histograma
y color de barra azul. Poner un título al eje x para diferenciar que son los números que se van a repetir.
"""
eje_x = [str(k) for k in sorted(resumen)]
eje_y = [resumen[k] for k in sorted(resumen)]

plt.bar(x=eje_x, height= eje_y,width=0.75,edgecolor="black")
plt.xlabel('Números elegidos')
plt.title('Histograma')
plt.show()
"""
2.3 crea un diagrama de barras para el vector fusionado
con los 3 elementos más repetidos, si los hay, sin contar la moda.
"""
comunes = cuenta.most_common(4)

plt.bar([val[0]for val in comunes[1:5]],[val[1]for val in comunes[1:5]])
plt.show()

"Top3 sin la moda"

top3 = cuenta.most_common(4)[1:]

eje_x = [str(d[0]) for d in sorted(top3)]
eje_y = [d[1] for d in sorted(top3)]

plt.bar(x = eje_x, height = eje_y, width=0.75,edgecolor='black')
plt.show()


"""
BLOQUE 3: CARGA DE FICHEROS Y MODELIZACION.

Tenemos un fichero llamado data_examen_sanit del cual queremos realizar una modelizacion mediante
un arbol de decision y una regresión logística.
La columna objetivo de la tabla se llama "target", y el resto de columnas actuan como input para predecir el Target

#3.1 Cargue dicho fichero y prepara los datos para la modelización.
#Con una proporcion 70-30 para los conjuntos train y test.
#Obtén los estadisticos más importantes para las variables del conjunto de datos
"""
"Importamos los datos"
data = pd.read_csv("/home/domingo/Desktop/Python/data_examen_sanit.txt", sep = "\t", header=0)
print(data.head())
print(data.describe())


"Seleccionamos el target y las variables predictoras"

X = data.values[:, 1:data.shape[1]] 
Y = data.values[:, 0] 
  

"Dividimos el dataset en 0.7 para train y el resto para test"
X_train, X_test, y_train, y_test = train_test_split(  
X, Y, test_size = 0.3, random_state = 100) 


"""
3.2 Entrene y evalue los modelos. Arbol y reg. logística
"""
"Creamos árbol basado en el Gini o impureza"

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5, splitter = "best")
clf_fit = clf_gini.fit(X_train, y_train) 
clf_pred = clf_gini.predict(X_test)
print(clf_pred[0:10])


print("Confusion Matrix Gini Tree: ", confusion_matrix(y_test, clf_pred)) 
      
print ("Accuracy Gini Tree: ", accuracy_score(y_test,clf_pred)*100) 
      
print("Report Gini Tree: ", classification_report(y_test, clf_pred)) 


"Creamos modelo de regresion logistica"

logreg = LogisticRegression(solver ='lbfgs',max_iter=1000)
logreg_fit = logreg.fit(X_train,y_train)
logreg_pred = logreg.predict(X_test)
logreg_pred_proba = logreg.predict_proba(X_test)
print('Accuracy of logistic regression classifier is: {:.2f}'.format(logreg.score(X_test,y_test)))
print("Confusion Matrix logistic regression: ", confusion_matrix(y_test, logreg_pred)) 
      
print ("Accuracy logistic regression: ", accuracy_score(y_test,logreg_pred)*100) 
      
print("Report logistic regression: ", classification_report(y_test, logreg_pred)) 

print(logreg_pred)

"""
3.3 Proponga un método para comparar ambos modelos (ej: Curva ROC)
"""
"Elaboramos curva ROC para el arbol con criterio gini"
logic_roc_auc =roc_auc_score(y_test,clf_pred)
fpr,tpr, thresholds = roc_curve(y_test, clf_pred)
plt.figure()
plt.plot(fpr, tpr, label='Decision tree (area = %0.2f)' % logic_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Gini Tree')
plt.legend(loc="lower right")
plt.show()


"Elaboramos curva ROC para regresión logística"

logic_roc_auc =roc_auc_score(y_test,logreg_pred)
fpr,tpr, thresholds = roc_curve(y_test, logreg_pred_proba[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logic_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for log. regression')
plt.legend(loc="lower right")
plt.show()