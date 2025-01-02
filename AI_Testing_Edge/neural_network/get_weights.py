import keras
from pathlib import Path

model = keras.models.load_model(Path(__file__).parent / 'model.keras')

tables = []

with open(Path(__file__).parent / "weights.c", "w") as f:
    f.write('#include "weights.h"\n\n')
    for layer in model.layers:
        if "input" not in layer.get_config()['name']:
            weights, bias = layer.get_weights()
            weights = weights.tolist()
            bias = bias.tolist()
            name, activation = layer.get_config()['name'], layer.get_config()['activation']
            table_name = f"float {name}_weights[{len(weights)}][{len(weights[0])}]"
            tables.append(table_name)
            table_str = table_name  + " = {\n"
            for row in weights:
                row_str = "{"
                for item in row:
                    row_str = row_str + f"{item}, "
                row_str = row_str.rstrip(",") + "},\n"
                table_str = table_str + row_str
            table_str = table_str.rstrip(",") + "\n};\n\n"
            f.write(table_str)

            table_name = f"float {name}_bias[{len(bias)}]"
            tables.append(table_name)
            table_str = table_name  + " = { "
            for item in bias:
                table_str = table_str + f"{item}, "
            table_str = table_str.rstrip(",") + " };\n\n"
            f.write(table_str)

with open(Path(__file__).parent / "weights.h", "w") as f:
    f.write(
        """
#ifndef WEIGHTS_H
#define WEITGHS_H

""")
    for table in tables:
        f.write(f"extern {table}\n")
    f.write(
"""
#endif
"""
    )