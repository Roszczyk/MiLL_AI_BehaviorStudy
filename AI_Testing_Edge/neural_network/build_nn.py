import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
import numpy as np


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

    Xs = []
    Ys = []

    for d in data:
        Y = d.pop(0)
        Ys.append(Y)
        Xs.append(d)

    Xs = np.array(Xs, dtype=float)
    Ys = np.array(Ys, dtype=float)
    Ys = keras.utils.to_categorical(Ys, 4)

    X_train, X_test, y_train, y_test = train_test_split(Xs, Ys, random_state=1)

    model_input = keras.Input(shape = (len(Xs[0]),))
    x = keras.layers.Dense(120, activation='relu')(model_input)
    x = keras.layers.Dense(40, activation='sigmoid')(x)
    outputs = keras.layers.Dense(4, activation='softmax')(x)
    model = keras.Model(inputs=model_input, outputs=outputs)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=1000, batch_size=32)

    preds = model.predict(X_test)
    preds = np.argmax(preds, axis=1)
    print(preds)

    print("accuracy: ", accuracy_score(np.argmax(y_test, axis=1), preds))

    model.save(Path(__file__).parent / "model.keras")