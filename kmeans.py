import math
import csv
import random

def distance(obj1, obj2, attributes):
    dist = 0
    for attr in attributes:
        dist += (obj1[attr] - obj2[attr]) ** 2
    return math.sqrt(dist)

# Entrada:
# 	k: Número de clústeres,
# 	D: Dataset compuesto de n objetos
def kmeans(k, D, attributes):            
# Método:
# 	1. Escoger k elementos de D como centros iniciales de los clústeres
    centroids = []
    new_centroids = [obj for obj in random.choices(D, k=k)]
    print(new_centroids)

# 	2. REPETIR
# 	3.	(re)asignar cada objeto al clúster, de acuerdo a la
#           "distancia" del objeto hacia el centro de cada clúster
    while (new_centroids != centroids):
        centroids = new_centroids
        clusters = [[] for i in range(k)]
        for obj_index, obj in enumerate(D):
            minimum = math.inf
            minimum_index = None
            for cluster_index, centroid in enumerate(centroids):
                _distance = distance(centroid, obj, attributes)
                if _distance < minimum:
                    minimum_index = cluster_index
                    minimum = _distance
            clusters[minimum_index].append(obj_index)
        
        # print(clusters)
    # 	4.	actualiza el centro de cada clúster, en base a la nueva
    # 			composición del clúster
        new_centroids = []
        for i in range(k):
            if len(clusters[i]) == 0:
                continue
            centroid = {}
            for obj_index in clusters[i]:
                obj = D[obj_index]
                for attr in attributes:
                    centroid[attr] = centroid.get(attr, 0) + obj[attr]
            for attr in attributes:
                centroid[attr] /= len(clusters[i])
            new_centroids.append(centroid)
        print(new_centroids)

# 	5. HASTA que se cumpla criterio de terminación
    return centroids

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
