import sys
import os 
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

print(pred)
print(y_test)
print(accuracy)

tree_capture = tree.tree_

print(tree_capture)

print(tree_capture.max_depth)

plot_tree(tree)
print(export_text(tree))

def export_tree(tree, feature_names=None):
    tree_structure = []
    index = []
    condition = []
    def recurse(node, level=0):
        if tree.feature[node] != -2:
            tree_structure.append({
                "level": level,
                "index": int(tree.feature[node]),
                "threshold": float(tree.threshold[node]),
                "left": tree.children_left[node],
                "right": tree.children_right[node],
            })
            index.append(int(tree.feature[node]))
            recurse(tree.children_left[node], level + 1)
            recurse(tree.children_right[node], level + 1)
        else:
            tree_structure.append({
                "level": level,
                "result": float(tree.value[node][0][0])
            })
    recurse(0)
    print(index)
    return tree_structure

tree_data = export_tree(tree_capture)

print(tree_data)

# def tree_to_c(tree_data):
#     c_code = []
#     for i, node in enumerate(tree_data):
#         if "result" in node:  # Liść
#             c_code.append(f"{{.level={node['level']}, .result={node['result']}}}")
#         else:  # Węzeł wewnętrzny
#             c_code.append(
#                 f"{{.level={node['level']}, .index={node['index']}, .threshold={node['threshold']}, "
#                 f".below={node['left']}, .over={node['right']}}}"
#             )
#     return "TreeNode nodes[] = {" + ",\n".join(c_code) + "};"

# c_tree_code = tree_to_c(tree_data)
# print(c_tree_code)
# print(tree_data)

