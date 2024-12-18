import sys
import os 
from pathlib import Path

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

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
        "wypoczyn" : 1,
        "integr" : 2,
        "integ" : 2,
        "suzbowy" : 3,
        "sluzbowy" : 3,
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

regressor = DecisionTreeRegressor(max_depth=3, random_state=42)
regressor.fit(X_train, y_train)

tree = regressor.tree_

def export_tree(tree, feature_names=None):
    tree_structure = []
    def recurse(node, level=0):
        if tree.feature[node] != -2:
            tree_structure.append({
                "level": level,
                "index": int(tree.feature[node]),
                "threshold": float(tree.threshold[node]),
                "left": tree.children_left[node],
                "right": tree.children_right[node],
            })
            recurse(tree.children_left[node], level + 1)
            recurse(tree.children_right[node], level + 1)
        else:
            tree_structure.append({
                "level": level,
                "result": float(tree.value[node][0][0])
            })
    recurse(0)
    return tree_structure

tree_data = export_tree(tree)

def tree_to_c(tree_data):
    c_code = []
    for i, node in enumerate(tree_data):
        if "result" in node:  # Liść
            c_code.append(f"{{.level={node['level']}, .result={node['result']}}}")
        else:  # Węzeł wewnętrzny
            c_code.append(
                f"{{.level={node['level']}, .index={node['index']}, .threshold={node['threshold']}, "
                f".below={node['left']}, .over={node['right']}}}"
            )
    return "TreeNode nodes[] = {" + ",\n".join(c_code) + "};"

c_tree_code = tree_to_c(tree_data)
print(c_tree_code)
print(tree_data)

