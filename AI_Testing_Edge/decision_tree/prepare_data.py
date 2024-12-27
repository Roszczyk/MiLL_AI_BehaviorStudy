from build_tree import init_file, strings_to_ints

from pathlib import Path


if __name__ == "__main__":
    path_to_data = Path(__file__).parent.parent.parent / "ML_visit_purpose/prepared_data.csv"
    data = init_file(path_to_data)

    with open(Path(__file__).parent / "data.c", "w") as data_file:
        data_file.write('#include "data.h"\n\n')
        data_file.write("float data_array[DATA_ROWS][DATA_ITEMS] = {\n")
        known = False
        for row in data:
            row_string = "    {"
            row.pop(0)
            if known != True:
                print(f"DATA ROW = {len(row)}")
                known = True
            for item in row:
                row_string = row_string + f"{item},"
            row_string = row_string.rstrip(",") + "},\n"
            data_file.write(row_string)
        data_file.write("};")
        print("DATA VOLUME = ", len(data))

    with open(Path(__file__).parent / "data.h", "w") as data_file:
        data_file.write(
            """
#ifndef DATA_H\n\
#define DATA_H\n\n\
#define DATA_ITEMS 11\n\
#define DATA_ROWS 102\n\n\
extern float data_array[DATA_ROWS][DATA_ITEMS];\n\n\
#endif\n
            """
        )