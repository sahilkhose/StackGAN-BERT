'''
Displays the 10 annotations and the corresponding picture.
'''
import config 

import numpy as np 
import os
import matplotlib.pyplot as plt 
import pickle
import torch

from PIL import Image
from scipy.spatial.distance import cosine
from tqdm import tqdm
print("__"*80)
print("Imports finished")
print("__"*80)


def display_specific(bird_type_no=0, file_no=0, file_idx=None):
    """
    Prints annotations and displays images of a specific bird
    """
    bird_type = sorted(os.listdir(config.IMAGE_DIR))[bird_type_no]
    file = sorted(os.listdir(os.path.join(config.ANNOTATIONS, bird_type)))[file_no]

    if file_idx is None:
        filename = os.path.join(bird_type, file)
    else:
        filenames = np.array(pickle.load(open("../data/birds/train/filenames.pickle", "rb"), encoding='latin1'))
        filename = filenames[file_idx]
        filename += ".txt"

    print(f"\nFile: {filename}\n")

    text = open(os.path.join(config.ANNOTATIONS, filename), "r").read().split('\n')[:-1]
    [print(f"{idx}: {line}") for idx, line in enumerate(text)]
    filename = filename.replace(".txt", ".jpg")
    plt.imshow(plt.imread(os.path.join(config.IMAGE_DIR, filename)))
    plt.show()

def compare_bert_emb(file_1, file_2, emb_no=0):
    emb_1 = torch.load(os.path.join(config.ANNOTATION_EMB, file_1, f"{emb_no}.pt"), map_location="cpu")
    emb_2 = torch.load(os.path.join(config.ANNOTATION_EMB, file_2, f"{emb_no}.pt"), map_location="cpu")
    # print(emb_1.shape) # (1, 768)

    bert_sim = 1 - cosine(emb_1, emb_2)
    print(f"cosine similarity bert emb: {bert_sim:.2f}")

def compare_cnn_emb(emb_idx_1, emb_idx_2, emb_no=0):
    embeddings = np.array(pickle.load(open("../data/birds/train/char-CNN-RNN-embeddings.pickle", "rb"), encoding='latin1'))
    # print(embeddings.shape) # (8855, 10, 1024)
    
    cnn_sim = 1 - cosine(embeddings[emb_idx_1][emb_no], embeddings[emb_idx_2][emb_no])
    print(f"cosine similarity cnn embs: {cnn_sim:.2f}")

def compare_embedding_quality(emb_idx_1=0, emb_idx_2=1, emb_no=0):
    ###* Filenames to fetch embs:
    filenames = np.array(pickle.load(open("../data/birds/train/filenames.pickle", "rb"), encoding='latin1'))
    # print(filenames.shape)  # (8855, )

    ###* File paths:
    file_1 = filenames[emb_idx_1]
    file_2 = filenames[emb_idx_2]

    print(f"File 1: {file_1}")
    print(f"File 2: {file_2}\n")

    ###* Annotations:
    text1 = open(os.path.join(config.ANNOTATIONS, file_1+".txt"), "r").read().split('\n')[:-1]
    text2 = open(os.path.join(config.ANNOTATIONS, file_2+".txt"), "r").read().split('\n')[:-1]
    print("Annotation 1: ", text1[emb_no])
    print("Annotation 2: ", text2[emb_no])
    print()

    ###* Cosine similarity:
    compare_cnn_emb(emb_idx_1, emb_idx_2, emb_no=0)
    compare_bert_emb(file_1, file_2, emb_no=0)

    ###* Display images:
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(plt.imread(os.path.join(config.IMAGE_DIR, file_1 + ".jpg")))
    fig.add_subplot(1, 2, 2)
    plt.imshow(plt.imread(os.path.join(config.IMAGE_DIR, file_2 + ".jpg")))
    # plt.show()

def check_model(file_idx, model):
    import sys
    sys.path.insert(1, "../../src/")
    import layers
    emb_no = 0
    ###* load the models
    netG = layers.Stage1Generator().cuda()
    netG.load_state_dict(torch.load(model))
    netG.eval()
    with torch.no_grad():
        ###* load the embeddings
        filenames = np.array(pickle.load(open("../data/birds/train/filenames.pickle", "rb"), encoding='latin1'))
        file_name = filenames[file_idx]
        emb = torch.load(os.path.join(config.ANNOTATION_EMB, file_name, f"{emb_no}.pt"))

        ###* Forward pass
        print(emb.shape)  # (1, 768)
        noise = torch.autograd.Variable(torch.FloatTensor(1, 100)).cuda()
        noise.data.normal_(0, 1)
        _, fake_image, mu, logvar = netG(emb, noise)
        fake_image = fake_image.squeeze(0)
        print(fake_image.shape)  #(3, 64, 64)

        im_save(fake_image, count=0)
    return fake_image

def im_save(fake_img, count=0):
    save_name = f"{count}.png"  
    im = fake_img.cpu().numpy()
    im = (im + 1.0) * 127.5
    im = im.astype(np.uint8)
    # print("im", im.shape)
    im = np.transpose(im, (1, 2, 0))
    # print("im", im.shape)
    im = Image.fromarray(im)
    im.save(save_name)

if __name__ == "__main__":
    # display_specific(bird_type_no=0, file_no=0)  # old method
    display_specific(file_idx=14)  # new method

    print("__"*80)
    # compare_embedding_quality(emb_idx_1=0, emb_idx_2=1, emb_no=0)
    ###* emb_idx < 8855, emb_no < 10


    print("__"*80)
    check_model(file_idx=14,
                model="../../old_outputs/output-3/model/netG_epoch_110.pth")

    plt.show()