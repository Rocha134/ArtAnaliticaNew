import math
import csv
import random

"""
== distancia euclidiana ==
Entrada:
    obj1, obj2: Objetos a regrupar (diccionarios)
    attributes: Lista de attributos (strings) a considerar para el cálculo de la distancia  
"""
def distance(obj1, obj2, attributes):
    return 0

"""
== K-Means ==
Entrada:
    k: Número de clústeres,
	D: Dataset compuesto de n objetos
"""
def kmeans(k, D, attributes):
# Método:
# 	1. Escoger k elementos de D como centros iniciales de los clústeres
# 	2. REPETIR
# 	3.	(re)asignar cada objeto al clúster, de acuerdo a la
#           "distancia" del objeto hacia el centro de cada clúster
# 	4.	actualiza el centro de cada clúster, en base a la nueva
# 			composición del clúster
# 	5. HASTA que se cumpla criterio de terminación
    return []

if __name__ == "__main__":
    file = open("Mall_Customers.csv", newline='')
    dataset = []
    for row in csv.DictReader(file):
        nrow = {}
        for key, value in row.items():
            if key == "Genre":
                nrow[key] = value
            else:
                nrow[key] = float(value)
        dataset.append(nrow)
    kmeans(3, dataset, ["Annual_Income_(k$)", "Spending_Score"])
