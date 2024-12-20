import sys
import os 
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def initFile(fileName): #odczytanie danych z pliku i stworzenie obiektów danych
    path_to_data = Path(__file__).parent.parent.parent / fileName
    file=open(path_to_data)
    dataArray=[]
    file_text=file.read()
    file_lines=file_text.split("\n")
    for i in range(len(file_lines)-1):
        file_line=file_lines[i]
        line_array=file_line.split(",")
        line_array.pop(0)
        dataArray.append(line_array)
    file.close()
    dataArray.pop(0)
    return dataArray

def strings_to_ints(data):
    translator = dict({
        "wypocz" : 1,
        "integr" : 2,
        "praca" : 3,
        "szkole" : 4
    })
    for i in range(len(data)):
        data[i][0] = translator[data[i][0]]
    return data

data = initFile("ML_visit_purpose/prepared_data.csv")
data = strings_to_ints(data)

Xs = []
Ys = []
for d in data:
    Y = d.pop(0)
    Ys.append(Y)
    Xs.append(d)

X_train, X_test, y_train, y_test = train_test_split(Xs, Ys, random_state=1)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, pred)

tree_capture = tree.tree_

DEPTH = tree_capture.max_depth

def export_tree(tree, feature_names=None):
    tree_structure = []
    index = [[0] * 2**(DEPTH-1) for _ in range(DEPTH)]
    condition = [[0] * 2**(DEPTH-1) for _ in range(DEPTH)]
    results = [0] * 2**DEPTH
    def recurse(node, level = 0, rowNumber = 0):
        if tree.feature[node] != -2:
            tree_structure.append({
                "level": level,
                "index": int(tree.feature[node]),
                "threshold": float(tree.threshold[node]),
                "left": tree.children_left[node],
                "right": tree.children_right[node],
            })
            index[level][rowNumber] = int(tree.feature[node])
            condition[level][rowNumber] =  float(tree.threshold[node])
            recurse(tree.children_left[node], level + 1, rowNumber = rowNumber * 2)
            recurse(tree.children_right[node], level + 1, rowNumber = rowNumber * 2 + 1)
        else:
            tree_structure.append({
                "level": level,
                "result": float(tree.value[node][0][0])
            })
            for i in range(rowNumber * 2**(DEPTH-level), (rowNumber + 1) * 2**(DEPTH-level)):
                print(tree.value[node][0][0])
                results[i] = tree.value[node][0][0]
            
    recurse(0)
    return tree_structure, index, condition, results

tree_data, index_array, condition_array, results_array = export_tree(tree_capture)

def tree_to_c(index, condition, results):
    all_index_string = "{"
    for row in index:
        row_string = "{"
        for i in row:
            row_string = row_string + f"{i},"
        row_string = row_string.rstrip(",") + "}"
        all_index_string = all_index_string + f"{row_string},"
    all_index_string = all_index_string.rstrip(",") + "}"

    
    all_condition_string = "{"
    for row in condition:
        row_string = "{"
        for i in row:
            row_string = row_string + f"{i},"
        row_string = row_string.rstrip(",") + "}"
        all_condition_string = all_condition_string + f"{row_string},"
    all_condition_string = all_condition_string.rstrip(",") + "}"

    all_results_string = "{"
    for i in results:
        all_results_string = all_results_string + f"{int(i)},"
    all_results_string = all_results_string.rstrip(",") + "}"

    return all_index_string, all_condition_string, all_results_string

print(len(results_array))
index_array, condition_array, results_array = tree_to_c(index_array, condition_array, results_array)


print("INDEX:\n", index_array)
print("CONDITIONS:\n", condition_array)
print("RESULTS:\n", results_array)
print("DEPTH: ", DEPTH)