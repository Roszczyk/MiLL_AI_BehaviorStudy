import numpy as np
from pathlib import Path

def init_file(file_path):
    file=open(file_path)
    data_array=[]
    file_text=file.read()
    file_lines=file_text.split("\n")
    for i in range(len(file_lines)-1):
        file_line=file_lines[i]
        line_array=file_line.split(",")
        line_array.pop(0)
        data_array.append(line_array)
    file.close()
    data_array.pop(0)
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


if __name__ == "__main__":
    path_to_data = Path(__file__).parent.parent.parent / "ML_visit_purpose/prepared_data.csv"
    data = init_file(path_to_data)
    data = strings_to_ints(data)

    # Dane treningowe
    X_train = np.array([[1.5, 2.3], [1.8, 2.5], [3.2, 4.1], [3.8, 4.3]])
    y_train = np.array([0, 0, 1, 1])  # Klasy: 0 i 1

    # Obliczanie priors
    classes, class_counts = np.unique(y_train, return_counts=True)
    priors = class_counts / len(y_train)

    # Obliczanie średnich i wariancji
    means = np.array([X_train[y_train == c].mean(axis=0) for c in classes])
    variances = np.array([X_train[y_train == c].var(axis=0) for c in classes])

    # Funkcja prawdopodobieństwa warunkowego
    def gaussian_likelihood(x, mean, var):
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))

    # Predykcja
    def predict(X):
        posteriors = []
        for c_idx, c in enumerate(classes):
            prior = np.log(priors[c_idx])  # Log-prior dla stabilności numerycznej
            likelihood = np.sum(np.log(gaussian_likelihood(X, means[c_idx], variances[c_idx])))
            posteriors.append(prior + likelihood)
        return classes[np.argmax(posteriors)]

    # Testowanie
    X_test = np.array([2.0, 3.0])
    predicted_class = predict(X_test)
    print(f"Predicted class: {predicted_class}")