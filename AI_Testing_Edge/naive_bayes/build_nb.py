import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score


def init_file(file_path):
    with open(file_path) as file:
        data_array = [
            line.strip().split(",")[1:]
            for line in file.readlines()[1:-1]
        ]
    return data_array


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


def prepare_x_y(data):
    Xs, Ys = [], []
    for d in data:
        Ys.append(d.pop(0))
        Xs.append([float(i) for i in d]) 
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    return Xs, Ys


def gaussian_likelihood(x, mean, var):
    var = np.maximum(var, 1e-6)
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))


def predict(X, classes, priors, means, variances):
    posteriors = []
    for c_idx, c in enumerate(classes):
        prior = np.log(priors[c_idx])
        cus = np.log(gaussian_likelihood(X, means[c_idx], variances[c_idx]))
        print("cus: ",cus)
        likelihood = np.sum(cus)
        posteriors.append(prior + likelihood)
    return classes[np.argmax(posteriors)]

def save_for_c(priors, means, variances):
    with open(Path(__file__).parent / "config.c", "w") as file:
        file.write('#include "config.h"\n\n')
        row_string = f"const float priors[{len(priors)}] =" + " { "
        for p in priors:
            row_string = row_string + f"{p}, "
        row_string = row_string.rstrip(",") + " };\n\n"
        file.write(row_string)
        table_string = f"const float means[{len(means)}][{len(means[0])}] = " + "{ \n"
        for row in means:
            row_string = "  { "
            for m in row:
                row_string = row_string + f"{m}, "
            row_string = row_string.rstrip(",") + " },\n"
            table_string = table_string + row_string
        table_string = table_string.rstrip(",\n") + "\n};\n\n"
        file.write(table_string)
        table_string = f"const float variances[{len(variances)}][{len(variances[0])}] = " + "{ \n"
        for row in variances:
            row_string = "  { "
            for v in row:
                row_string = row_string + f"{v}, "
            row_string = row_string.rstrip(",") + " },\n"
            table_string = table_string + row_string
        table_string = table_string.rstrip(",\n") + "\n};\n\n"
        file.write(table_string)

    with open(Path(__file__).parent / "config.h", "w") as file:
        file.write("#ifndef CONFIG_H\n#define CONFIG_H\n\n")
        file.write("#define CLASSES 4\n\n")
        file.write(f"extern const float priors[{len(priors)}];\n")
        file.write(f"extern const float means[{len(means)}][{len(means[0])}];\n")
        file.write(f"extern const float variances[{len(variances)}][{len(variances[0])}];\n")
        file.write("\n#endif")



if __name__ == "__main__":
    path_to_data = Path(__file__).parent.parent.parent / "ML_visit_purpose/prepared_data.csv"
    data = init_file(path_to_data)
    data = strings_to_ints(data)
    
    X_train, y_train = prepare_x_y(data)
    
    classes = np.unique(y_train)
    priors = np.array([np.sum(y_train == c) for c in classes]) / len(y_train)
    means = np.array([X_train[y_train == c].mean(axis=0) for c in classes])
    variances = np.array([X_train[y_train == c].var(axis=0) for c in classes])

    results = [int(predict(row, classes, priors, means, variances)) for row in X_train]

    print(f"Predicted classes: {results}")
    print(f"accuracy: {accuracy_score(results, y_train)}")

    save_for_c(priors, means, variances)