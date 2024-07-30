consolidated_labels = {
        # Abiotic
        "abiotic_aircraft": ["aircraft", "airplane", "airplanes", "loud thing (plane?)", "plane", "helicopter"],
        "abiotic_transient": ["raindrops", "water drops", "droplets", "raindrop", "drop", "droplet", "drip", "crackle", "dripping", "rapping on wood", "tap sound", "wood sound", "loud snap sound","dri","drippin","drips","loud drip on wood","popping noise","tapping","tapping sound","snap"],
        "abiotic_vegetation": ["branch", "branch rustling", "branches rustling", "branches snapping","ground  rustling", "ground rustling", "leaves rustling", "rustling", "tree creaking", "branches", "twig rustling", "leaves","branch cracking","cracking", "creaking","stick breaking","twig creaking","wood cracking","wood creaking","wooden rap","wooden soudn","wooden sound","crackling","vegetation","tree snap"],
        "abiotic_rain": ["rain","loud rain","rain "],
        "abiotic_wind": ["wind", " wind"],
        "abiotic_water": ["river", "water"],
        "abiotic_logging": ["chainsaw", "chainsaw revving", "chaindaw", "chainsa","chinsaw"],
        "abiotic_vehicle": ["machinery", "machine", "vehicle", "vehicle backing up noise", "truck reverse sound", "vehicle?","engine","machine banging","vehicle backing up", "car", "truck"],
        "abiotic_other": ["noise", "other", "random noise", "electricity", "high pitch buzzing", "thunder", "rain/wind", "buzzing sound", "buzz", "buzzing", "static sound", "interesting sound", "mystery noise", "static", "banging", "clink noise", "crashing sound","rumbling","thumping sound","bang","hum","really loud noise","loud noise","thunk"],
        # Biotic
        "anuran": ["frog", "frogs", "frog chirping"],
        "coyote": ["coyotes howling"],
        "dog": ["dogs"],
        "insect": ["bee"],
        "biotic_other": ["animal","wingbeats","bird flapping"],
        "unknown": ["unknown sparrow"],
        "0": [
            "not_target", "not-bird", "not bird", "not'species",
            "golden-crowned sparrow","sparrow","sparriw","sparrw","common loon","finch","grouse","hummingbird","kinglet","warbler","woodpecker"
            ], # 0 indicates a predicted species is NOT present
        # Other
        "truncation": ["truncated"],
}

print(consolidated_labels.keys())