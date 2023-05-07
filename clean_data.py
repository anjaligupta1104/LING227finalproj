import os

def filter_chi_lines(input_dir, output_dir):

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".txt"):
            input_file_path = os.path.join(input_dir, file_name)
            with open(input_file_path, "r", encoding="utf-8") as input_file:
                lines = input_file.readlines()

            filtered_lines = [line[5:].lstrip() for line in lines if line.startswith("*CHI:")]

            output_file_path = os.path.join(output_dir, file_name)
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                output_file.writelines(filtered_lines)

# Example usage
input_file_path = "./Gillam/text_to_clean"
output_file_path = "./data"
filter_chi_lines(input_file_path, output_file_path)