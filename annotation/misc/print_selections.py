import os

def process_txt_file(file_path, all_unique_species):
    unique_species = set()

    with open(file_path, 'r') as file:
        # Skip the header line
        next(file)
        
        for line in file:
            # Split the line into columns
            columns = line.strip().split('\t')

            # Extract species information (assuming species is in the last column)
            species = columns[-1]

            # Add the species to the set
            unique_species.add(species)
            if species not in all_unique_species:
                all_unique_species.add(species)

    return unique_species

def main(directory_path):
    all_unique_species = set()

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)

            unique_species = process_txt_file(file_path, all_unique_species)

            print(f"{filename}:")
            print(f"{len(unique_species)} species:", ', '.join(unique_species))

    print(f"\n{len(all_unique_species)} total unique species:", ', '.join(all_unique_species))

if __name__ == "__main__":
    directory_path = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/Annotation/Data/_Reviewer/SMA00404_20230518/' # Change this to the path of your directory
    main(directory_path)
    
