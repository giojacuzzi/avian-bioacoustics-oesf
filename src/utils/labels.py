import pandas as pd
import Levenshtein

def get_species_classes():
    species_classes = pd.read_csv('/Users/giojacuzzi/repos/avian-bioacoustics-oesf/src/classification/species_list/species_list_OESF.txt', header=None) # Get list of all species
    species_classes = [name.split('_')[1].lower() for name in species_classes[0]]
    return species_classes

def correct_typo(word, min_distance=3):
    closest_words = []
    correct_labels = get_species_classes()
    correct_labels.append('unknown')
    correct_labels.append('not_target')
    for correct_word in correct_labels:
        distance = Levenshtein.distance(word, correct_word)
        if distance <= min_distance:
            closest_words.append(correct_word)
    if len(closest_words) == 0:
        return word
    else:
        return closest_words[0]


def clean_label(label):

    # Correct any typos
    label = correct_typo(label)

    # Manually consolidate any other specific labels
    consolidated_labels = {
        # Abiotic
        "abiotic_aircraft": ["aircraft", "airplane", "airplanes", "loud thing (plane?)", "plane"],
        "abiotic_transient": ["raindrops", "water drops", "droplets", "raindrop", "drip", "crackle", "dripping", "rapping on wood", "tap sound", "wood sound", "loud snap sound"],
        "abiotic_vegetation": ["branch", "branch rustling", "branches rustling", "branches snapping","ground  rustling", "ground rustling", "leaves rustling", "rustling", "tree creaking", "branches", "twig rustling", "leaves"],
        "abiotic_rain": ["loud rain"],
        "abiotic_wind": ["wind"],
        "abiotic_water": ["river", "water"],
        "abiotic_other_anthropogenic": ["machinery", "vehicle", "vehicle backing up noise", "vehicle?", "chainsaw"],
        "abiotic_other": ["noise", "other", "random noise", "electricity", "high pitch buzzing", "thunder", "rain/wind", "buzzing sound", "buzz", "buzzing", "static sound", "interesting sound", "mystery noise", "static"],
        # Biotic
        "anuran": ["frog", "frog chirping"],
        "coyote": ["coyotes howling"],
        "dog": ["dogs"],
        "insect": ["bee"],
        "biotic_other": ["animal"],
        "0": ["not_target", "not-bird", "not bird", "not'species"], # 0 indicates a predicted species is NOT present
    }

    for correction, typos in consolidated_labels.items():
        if label in typos:
            return correction
    return label

# TEST
# print(clean_label('pacifi-slope flycather'))
# print(clean_label('unknoen'))
# print(clean_label('nonexistant label'))
# print(clean_label("macgillvary's warbler"))
