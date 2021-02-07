import numpy as np
import pandas as pd
import math
import random
from pandas import DataFrame

from matplotlib import pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import scale

# from TSP import Dynamic
# from TSP import Greedy

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=names)
df.head()

# unique function check to check the unique numbers of user id and item id
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
# print (str(n_users) + ' users')
# print (str(n_items) + ' items')

ratingsBinary = np.zeros((n_users, n_items))
ratings = np.zeros((n_users, n_items))

threshold = 3
# df.intertuples will run the code directly
for row in df.itertuples():
    # mark ratings that over 3 and 4 to 1; the rest of them will be set to 0.
    if row[3] > threshold:
        ratingsBinary[row[1]-1, row[2]-1] = 1
        ratings[row[1]-1, row[2]-1] = row[3]
# print(ratings)
# print(ratingsBinary)

# save binary contents to a csv file
df = DataFrame(ratingsBinary)
df.to_csv('binary_values.csv', index = False, header = False)

# randomly select 400 items from 0, 1000; replace = False (there is no back)
biclusters_num = 400
biclusters_rows = int(biclusters_num**0.5)
biclusters_cols = int(biclusters_num**0.5)
selection_index = np.random.choice(a = 1000, size= biclusters_num, replace = False)
# print(selection_index)

# read file from matlab which generates biclustering files
filename = "biclusters.csv"
cluster_no = 0

clusters_number = 1000

f = open(filename)

# create sub matrix to record biclustering
for i in range(0, clusters_number):

    # obtain the index of rows and cols separately
    rows = f.readline().split()
    cols = f.readline().split()

    # valid whether the bimax runned correctly or not
    i = np.zeros((len(rows), len(cols)))

    # put the ratings back to biclustering matrix to valid the performance
    row_count = 0
    for j in rows:
        col_count = 0

        for k in cols:
            i[row_count, col_count] = ratings[int(j) - 1, int(k) - 1]
            col_count += 1
        row_count += 1
        col_count = 0
    # print(i)
f.close()

# read file from matlab which generates biclustering files
filename = "biclusters.csv"
cluster_no = 0

clusters_number = 1000

f = open(filename)

# save the index of bicluster index
# data type = biclustername : [rows index][cols index] eg. 1:[[2,3,4][3,4,5]]
dict_clusters = {}
# save the detail ratings of the biclusters
# data type = arrayname : [detail ratings] eg. 1:[]
dict_clusters_ratings = {}

# create sub matrix to record biclustering
for i in range(0, clusters_number):
    # obtain the index of rows and cols separately
    dictname = str(i)

    rows = f.readline().split()
    cols = f.readline().split()

    # put user and item index into the dictionary -- dict_clusters
    dict_clusters[dictname] = [rows, cols]
f.close()
# print(dict_clusters['bicluster_999'])

# get the specific index of what selected
for i in selection_index:
    dictname = str(i)
    arrayname = str(i)

    rows = dict_clusters[dictname][0]
    cols = dict_clusters[dictname][1]

    a = np.zeros((len(rows), len(cols)))

    row_count = 0
    for j in rows:
        col_count = 0

        for k in cols:
            a[row_count, col_count] = ratings[int(j) - 1, int(k) - 1]
            col_count += 1
        row_count += 1
        col_count = 0
    # put array into the dictionary dict_clusters_ratings
    dict_clusters_ratings[arrayname] = a

# print(dict_clusters_ratings['array_1'])

# show the location of each compressed point
x = []
y = []

PCA_dict_clusters = dict_clusters_ratings.copy()

for i in PCA_dict_clusters.keys():
    # print(i)

    PCA_dict_clusters[i] = StandardScaler().fit_transform(PCA_dict_clusters[i])
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(PCA_dict_clusters[i])

    PCA_dict_clusters[i] = np.mean(np.abs(principalComponents), axis=0)
    # PCA_dict_clusters[i] = np.mean(np.power(principalComponents, 2), axis = 0)

    x.append(PCA_dict_clusters[i][0])
    y.append(PCA_dict_clusters[i][1])

x = np.array(x)
y = np.array(y)

plt.scatter(x,y, s = 30, alpha=0.3)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2 Component PCA by abs')

# plt.show()

# calculate the distance between A and B points;
# A : array, B : array
def eucliDist(A,B):
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))

# distance table to recorder the distance between two different data points
dist_matrix = np.zeros((biclusters_num, biclusters_num))
for i in range(biclusters_num):
    for j in range(biclusters_num):
        # assign value to dist
        dist_matrix[i,j] = eucliDist(PCA_dict_clusters[list(PCA_dict_clusters.keys())[i]]
                              , PCA_dict_clusters[list(PCA_dict_clusters.keys())[j]])
# print(PCA_dict_clusters.keys())

# dist_matrix
dist_matrix = dist_matrix.tolist()

# save weights for each length
path_length = []
# save the vertex that has been visited to prevent revisit again
path_vertexs = []
# real routes
path = []


def find_path(j, vertex_len):
    path_vertexs.append(j)
    row = dist_matrix[j]

    # copy_row: delete the vertex that has been visited --> prevent to operate it in the original rows directly
    copy_row = [value for value in row]

    walked_vertex = []

    # save the vertex that has been visited to walked vertex
    for i in path_vertexs:
        walked_vertex.append(copy_row[i])

    #  remove the vertex that has been visited in the copy_row
    for vertex in walked_vertex:
        copy_row.remove(vertex)

    # find the shortest value that never accessed in the row
    if len(path_vertexs) < vertex_len:
        min_e = min(copy_row)
        j = row.index(min_e)
        path_length.append(min_e)
        find_path(j, vertex_len)
    else:
        min_e = dist_matrix[j][0]
        path_length.append(min_e)
        path_vertexs.append(0)
    return path_vertexs, path_length


def print_path(vertexs, lengths):
    vertexs = [vertex + 1 for vertex in vertexs]
    for i, vertex in enumerate(vertexs):
        path.append(vertex)

        if i == len(dist_matrix):
            break

    # ("the smallest total value is：", sum(lengths))
    # print("path is：", path)


path_vertexs, path_length = find_path(0, len(dist_matrix))
print_path(path_vertexs, path_length)

# put the selected 400 biclusters into a new dict biclusters
# refactor name and index to make it easy to be found in the next stage
new_dict_biclusters = {}

k = 1
for i in selection_index:
    new_dict_biclusters[k] = dict_clusters[str(i)]
    k += 1
# print(new_dict_biclusters)

states = np.zeros((biclusters_rows, biclusters_cols))

# recorder the index of path array
k = 0

increment = range(1, int(biclusters_cols), 1)
decrement = range(int(biclusters_cols - 1), 0, -1)

states[0][0] = path[k]

for row in range(biclusters_rows):
    if row % 2 == 0:
        cols = increment
    elif row % 2 == 1:
        cols = decrement

    for col in cols:
        k += 1
        states[row][col] = path[k]

for j in range(biclusters_rows-1, 0, -1):
    k += 1
    states[j][0] = path[k]

# print(states)
# new_states =  np.zeros((biclusters_rows, biclusters_cols))
# for row in range(biclusters_rows):
#     for col in range(biclusters_cols):
#         new_states[row][col] = new_dict_biclusters[states[row][col]]


