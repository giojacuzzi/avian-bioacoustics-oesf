from birdnetlib.species import SpeciesList
import pandas as pd
import os

species = SpeciesList()
species_list = pd.DataFrame(species.return_list())
species_list['label'] = species_list['scientific_name'] + '_' + species_list['common_name']
print(species_list['label'])
pd.DataFrame.to_csv(species_list['label'], os.path.dirname(__file__) + '/_output/available_species.csv', index=False) 