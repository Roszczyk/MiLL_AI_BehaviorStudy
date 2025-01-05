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
        likelihood = np.sum(np.log(gaussian_likelihood(X, means[c_idx], variances[c_idx])))
        posteriors.append(prior + likelihood)
    return classes[np.argmax(posteriors)]


if __name__ == "__main__":
    path_to_data = Path(__file__).parent.parent.parent / "ML_visit_purpose/prepared_data.csv"
    data = init_file(path_to_data)
    data = strings_to_ints(data)
    
    X_train, y_train = prepare_x_y(data)
    
    print(f"Training on {len(X_train)} samples.")
    classes = np.unique(y_train)
    priors = np.array([np.sum(y_train == c) for c in classes]) / len(y_train)
    means = np.array([X_train[y_train == c].mean(axis=0) for c in classes])
    variances = np.array([X_train[y_train == c].var(axis=0) for c in classes])

    results = [int(predict(row, classes, priors, means, variances)) for row in X_train]

    print(f"Predicted classes: {results}")
    print(f"accuracy: {accuracy_score(results, y_train)}")
