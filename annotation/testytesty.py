
# Directory path containing the .txt files
directory_path = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/Annotation/Data/_Reviewer'
import os
from collections import defaultdict

# Function to parse a .txt file and extract unique species
def parse_txt_file(file_path):
    species_set = set()
    with open(file_path, 'r') as file:
        # Skip the header line
        next(file)
        for line in file:
            species = line.strip().split('\t')[-1]
            species_set.add(species)
    return species_set

# Function to recursively get all .txt files in subdirectories
def get_all_txt_files(directory_path):
    txt_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files

# Dictionary to store species and corresponding file paths
species_files = defaultdict(list)

# List to store file paths
file_paths = get_all_txt_files(directory_path)

# Iterate through all .txt files
for file_path in file_paths:
    species_set = parse_txt_file(file_path)
    species_files[file_path] = species_set

# Sort files based on the number of unique species
sorted_files = sorted(file_paths, key=lambda file_path: len(species_files[file_path]), reverse=True)

# Print files in the order of their number of unique species
print("Files ordered by the number of unique species:")
i = 1
for file_path in sorted_files:
    print(f"{os.path.basename(file_path)}: {len(species_files[file_path])} unique species ({i}): {species_files[file_path]}")
    i = i + 1

# Find all unique species across all files
all_species = set()
for species_set in species_files.values():
    all_species.update(species_set)

# Print all unique species
print("\nAll unique species across all files:")
for species in all_species:
    print(species)
