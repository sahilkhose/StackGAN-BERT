'''
Displays the 10 annotations and the corresponding picture.
'''
import config 

import numpy as np 
import os
import matplotlib.pyplot as plt 
import pickle
import torch

from scipy.spatial.distance import cosine
from tqdm import tqdm
print("__"*80)
print("Imports finished")
print("__"*80)


def annotations_image_show(bird_type, file):
    """
    Prints annotations, opens bird images
    """
    text = open(os.path.join(config.ANNOTATIONS, bird_type, file), "r").read().split('\n')[:-1]
    [print(f"{idx}: {line}") for idx, line in enumerate(text)]
    file = file.replace(".txt", ".jpg")
    plt.imshow(plt.imread(os.path.join(config.IMAGE_DIR, bird_type, file)))
    plt.show()

def display():
    """
    Prints annotations and displays images in a for loop
    """
    # for bird_type in tqdm(sorted(os.listdir(config.IMAGE_DIR)), total=len(os.listdir(config.IMAGE_DIR))):
    for bird_type in sorted(os.listdir(config.IMAGE_DIR)):
        for file in sorted(os.listdir(os.path.join(config.ANNOTATIONS, bird_type))):
            annotations_image_show(bird_type, file)
            break
        break

def display_specific(bird_type_no=0, file_no=0):
    """
    Prints annotations and displays images of a specific bird
    """
    bird_type = sorted(os.listdir(config.IMAGE_DIR))[bird_type_no]
    file = sorted(os.listdir(os.path.join(config.ANNOTATIONS, bird_type)))[file_no]
    annotations_image_show(bird_type, file)


def compare_new_bert(file_1, file_2, emb_no=0):
    emb_1 = torch.load(os.path.join(config.ANNOTATION_EMB,
                            file_1, f"{emb_no}.pt"), map_location="cpu")

    emb_2 = torch.load(os.path.join(config.ANNOTATION_EMB,
                                    file_2, f"{emb_no}.pt"), map_location="cpu")

    bert_sim = 1 - cosine(emb_1, emb_2)
    print(f"cosine similarity bert emb: {bert_sim:.2f}")

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(plt.imread(os.path.join(config.IMAGE_DIR, file_1 + ".jpg")))
    fig.add_subplot(1, 2, 2)
    plt.imshow(plt.imread(os.path.join(config.IMAGE_DIR, file_2 + ".jpg")))

    plt.show()


def compare_embedding_quality(emb_idx_1=0, emb_idx_2=1, emb_no=0):
    embeddings = np.array(pickle.load(open("../data/birds/train/char-CNN-RNN-embeddings.pickle", "rb"), 
                                            encoding='latin1'))
    # print(embeddings.shape) # (8855, 10, 1024)
    
    filenames = np.array(pickle.load(open("../data/birds/train/filenames.pickle", "rb"), 
                                            encoding='latin1'))
    # print(filenames.shape)  # (8855, )

    file_1 = filenames[emb_idx_1]
    file_2 = filenames[emb_idx_2]

    print(f"File 1: {file_1}")
    print(f"File 2: {file_2}\n")

    cnn_sim = 1 - cosine(embeddings[emb_idx_1][emb_no], embeddings[emb_idx_2][emb_no])

    print(f"cosine similarity cnn embs: {cnn_sim:.2f}")
    compare_new_bert(file_1, file_2, emb_no=0)

if __name__ == "__main__":
    # display()
    # display_specific(bird_type_no=1, file_no=0)


    compare_embedding_quality(emb_idx_1=0, emb_idx_2=1, emb_no= 0)
    # emb_idx < 8855, emb_no < 10
