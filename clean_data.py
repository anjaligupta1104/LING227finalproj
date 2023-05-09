import os

def filter_chi_lines(input_dir, output_dir):

    i = 1
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".cha"):
            input_file_path = os.path.join(input_dir, file_name)
            with open(input_file_path, "r", encoding="utf-8") as input_file:
                lines = input_file.readlines()

            filtered_lines = [line[5:].lstrip() for line in lines if line.startswith("*CHI:")]

            output_file_name = folder + "_" + child + "_" + str(i) + ".txt"
            output_file_path = os.path.join(output_dir, output_file_name)
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                output_file.writelines(filtered_lines)

            i += 1

# change input and output file paths to your own, although this should work as is

# Example usage
folder = "TD"
child = "11m"
input_file_path = "./Gillam/" + folder + "/" + child

if folder == "SLI":
    output_file_path = "./data/impaired"
else: output_file_path = "./data/not_impaired"

filter_chi_lines(input_file_path, output_file_path)