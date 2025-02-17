import pandas as pd
import Levenshtein

consolidated_labels = {
        # Abiotic
        "abiotic aircraft": ["abiotic_aircraft","aircraft", "airplane", "airplanes", "loud thing (plane?)", "plane", "helicopter"],
        "abiotic ambience": ["abiotic_ambience","ambient","ambience","branch", "branch rustling", "branches rustling", "branches snapping","ground  rustling", "ground rustling", "leaves rustling", "rustling", "tree creaking", "branches", "twig rustling", "leaves","branch cracking","cracking", "creaking","stick breaking","twig creaking","wood cracking","wood creaking","wooden rap","wooden soudn","wooden sound","crackling","vegetation","tree snap", "crackle", "rapping on wood", "tap sound", "wood sound", "loud snap sound","popping noise","tapping","tapping sound","snap"],
        "abiotic rain": ["abiotic_rain","rain","loud rain","rain "],
        "abiotic wind": ["abiotic_wind","wind", " wind"],
        "abiotic logging": ["abiotic_logging","logging","chainsaw", "chainsaw revving", "chaindaw", "chainsa","chinsaw"],
        "abiotic vehicle": ["abiotic_vehicle","abiotic_vehicle_reverse", "vehicle", "vehicle backing up noise", "truck reverse sound", "vehicle?","engine","vehicle backing up", "car", "truck"],
        "abiotic other": ["abiotic_other","noise", "other", "random noise", "electricity", "thunder", "rain/wind", "static sound", "interesting sound", "mystery noise", "static", "banging", "clink noise", "crashing sound","rumbling","thumping sound","bang","hum","really loud noise","loud noise","thunk","river", "water","raindrops", "water drops", "droplets", "raindrop", "drop", "droplet", "drip", "dripping","dri","drippin","drips","loud drip on wood","machinery", "machine","machine banging"],
        # Biotic
        "biotic anuran": ["biotic_anuran","anuran","frog", "frogs", "frog chirping"],
        "biotic coyote": [ "biotic_coyote","coyote","coyotes howling"],
        "biotic dog": ["biotic_dog","dog","dogs"],
        "biotic insect": ["biotic_insect","insect","bee","buzzing sound", "buzz", "buzzing", "high pitch buzzing"],
        "biotic other": ["biotic_other","animal","wingbeats","bird flapping","squirrel"],
        "not_target": [
            "0","not_target", "not-bird", "not bird", "not'species","unknown sparrow",
            "golden-crowned sparrow","sparrow","sparriw","sparrw","common loon","finch","grouse","hummingbird","kinglet","warbler","woodpecker"
        ], # 0 indicates a predicted species is NOT present
        # Other
        "artifact truncation": ["artifact_truncation","truncation","truncated"],
    }

def get_species_classes():
    species_classes = pd.read_csv('/Users/giojacuzzi/repos/avian-bioacoustics-oesf/src/classification/species_list/species_list_OESF.txt', header=None) # Get list of all species
    species_classes = [name.split('_')[1].lower() for name in species_classes[0]]
    return species_classes

def correct_typo(word, min_distance=2):
    closest_words = []
    correct_labels = get_species_classes()
    correct_labels.append(list(consolidated_labels.keys()))
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

    # Manually consolidate any specific labels
    for correction, typos in consolidated_labels.items():
        if label in typos:
            return correction

    # Correct any typos
    label = correct_typo(label)

    return label

# TEST
# print(clean_label('pacifi-slope flycather'))
# print(clean_label('unknoen'))
# print(clean_label('nonexistant label'))
# print(clean_label("macgillvary's warbler"))
# print(clean_label("song sparrow"))
# print(clean_label("Song sparrow"))
# print(clean_label("Song Sparrow"))
