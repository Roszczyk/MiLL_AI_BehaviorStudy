import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
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
        Xs.append(d)
    return train_test_split(np.array(Xs, dtype=float), np.array(Ys, dtype=int), random_state=1)


def gaussian_likelihood(x, mean, var):
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
    X_train, X_test, y_train, y_test = prepare_x_y(data)

    classes = np.unique(y_train)
    priors = np.array([np.sum(y_train == c) for c in classes]) / len(y_train)
    means = np.array([X_train[y_train == c].mean(axis=0) for c in classes])
    variances = np.array([X_train[y_train == c].var(axis=0) for c in classes])

    results = [int(predict(row, classes, priors, means, variances)) for row in X_test]
    print(f"Predicted classes: {results}")
