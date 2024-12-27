import sys
import os 
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


def initFile(file_path):
    file=open(file_path)
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
        "wypocz" : 0,
        "integr" : 1,
        "praca" : 2,
        "szkole" : 3
    })
    for i in range(len(data)):
        data[i][0] = translator[data[i][0]]
    return data


def export_tree(tree):
    tree_structure = []
    index = [[0] * 2**(DEPTH - 1) for _ in range(DEPTH)]
    condition = [[0] * 2**(DEPTH - 1) for _ in range(DEPTH)]
    results = [0] * 2**(DEPTH)
    print("Tree feature:", tree.feature)
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
            condition[level][rowNumber] = float(tree.threshold[node])
            print(level)
            recurse(tree.children_left[node], level + 1, rowNumber = rowNumber * 2)
            recurse(tree.children_right[node], level + 1, rowNumber = rowNumber * 2 + 1)
        else:
            tree_structure.append({
                "level": level,
                "result": float(tree.value[node][0][0])
            })
            for i in range(rowNumber * 2**(DEPTH-level), (rowNumber + 1) * 2**(DEPTH-level)):
                results[i] = np.argmax(tree.value[node][0])
    recurse(0)
    return tree_structure, index, condition, results

def tree_to_c(index, condition, results):
    all_index_string = "{\n"
    for row in index:
        row_string = "{"
        for i in row:
            row_string = row_string + f"{i},"
        row_string = row_string.rstrip(",") + "}"
        all_index_string = all_index_string + f"{row_string},\n"
    all_index_string = all_index_string.rstrip(",") + "\n}\n"

    all_condition_string = "{\n"
    for row in condition:
        row_string = "{"
        for i in row:
            row_string = row_string + f"{i},"
        row_string = row_string.rstrip(",") + "}"
        all_condition_string = all_condition_string + f"{row_string},\n"
    all_condition_string = all_condition_string.rstrip(",") + "\n}\n"

    all_results_string = "{"
    for i in results:
        all_results_string = all_results_string + f"{int(i)},"
    all_results_string = all_results_string.rstrip(",") + "}\n"

    return all_index_string, all_condition_string, all_results_string


def visualize_tree(tree):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))  # Rozmiar wykresu
    plot_tree(tree)
    plt.show()

if __name__ == "__main__":
    path_to_data = Path(__file__).parent.parent.parent / "ML_visit_purpose/prepared_data.csv"
    data = initFile(path_to_data)
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

    tree_data, index_array, condition_array, results_array = export_tree(tree_capture)
    index_array, condition_array, results_array = tree_to_c(index_array, condition_array, results_array)

    print("INDEX:\n", index_array)
    print("CONDITIONS:\n", condition_array)
    print("RESULTS:\n", results_array)
    print("DEPTH: ", DEPTH)

    single_vector = [
        4.0, 8.0, 28.669395973154366, 28.226168574812476, 27.78294117647059, 54.27523489932888, 
        54.1679115673115, 54.06058823529412, 0.005036419911470156, 0.034154420236290284, 0.03665182812583781
    ]
    prediction = tree.predict([single_vector])
    print("Prediction for single vector:", prediction)

    # visualize_tree(tree)