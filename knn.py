import numpy as np
import math
import pandas as pd
import operator

# import data csv

data = pd.read_csv("F:/Kuliah/Semester 7/Data Mining/Praktikum/2/data.csv")

data.head()
# print(data)
# untuk tes apakah file berhasil dibaca atau engga

##menghitung euclidian distance
def calc_euclid(data1, data2, length):
    distance = 0
    for x in range (length):
        distance += np.square(data1[x]-data2[x])
    
    return np.sqrt(distance)

##KNN Model
def knn (trainingSet, testingSet, k):

    distance = {}

    length = testingSet.shape[1]

    ##menghitung euclidean distance antara data training dengan
    ##data tes
    for x in range (len(trainingSet)):
        dist = calc_euclid (testingSet, trainingSet.iloc[x],length)

        distance[x] = dist[0]

    #Sorting hasil euclidean distance
    sorted_d = sorted(distance.items(), key=operator.itemgetter(1))
    print (sorted_d)

    neighbors = []


    #Mengekstrak nilai k
    for x in range (k):
        neighbors.append(sorted_d[x][0])
    print (neighbors)
    classVotes = {}

    ##Menghitung class paling banyak yang muncul
    for x in range (len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]

        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes [response] = 1
    ##sorting neighbor
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), 
    reverse =True)
    # print(sortedVotes)``
    return (sortedVotes[0][0],neighbors)

##Testing Set
testSet = [[1.52, 14, 2, 71, 7.8]]
test = pd.DataFrame(testSet)

##Menentukan nilai k
##Isi k dengan angka ganjil
k = 7

result,neigh = knn (data, test, k)

print (result)
print (neigh)
    





