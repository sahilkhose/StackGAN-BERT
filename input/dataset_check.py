import os
from tqdm import tqdm
IMAGE_DIR = "./CUB_200_2011/images"
TEXT_DIR = "./birds/text_c10"

# for bird_type in tqdm(sorted(os.listdir(IMAGE_DIR)), total=len(os.listdir(IMAGE_DIR))):
for bird_type in sorted(os.listdir(IMAGE_DIR)):
    for file in sorted(os.listdir(os.path.join(TEXT_DIR, bird_type))):
        text = open(os.path.join(TEXT_DIR, bird_type, file), "r").read().split('\n')[:-1]
        [print(f"{idx}: {line}") for idx, line in enumerate(text)]
        file = file.replace(".txt", ".jpg")
        os.system(f"xdg-open {os.path.join(IMAGE_DIR, bird_type, file)}")
        break
    break
