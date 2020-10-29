'''
Displays the 10 annotations and the corresponding picture.
'''
import config 

import os
from tqdm import tqdm


# for bird_type in tqdm(sorted(os.listdir(config.IMAGE_DIR)), total=len(os.listdir(config.IMAGE_DIR))):
for bird_type in sorted(os.listdir(config.IMAGE_DIR)):
    for file in sorted(os.listdir(os.path.join(config.ANNOTATIONS, bird_type))):
        text = open(os.path.join(config.ANNOTATIONS, bird_type, file), "r").read().split('\n')[:-1]
        [print(f"{idx}: {type(line)}") for idx, line in enumerate(text)]
        # file = file.replace(".txt", ".jpg")
        # os.system(f"xdg-open {os.path.join(config.IMAGE_DIR, bird_type, file)}")
        break
    break