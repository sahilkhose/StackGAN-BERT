"""Utility classes and methods.

Authors:
    Abhiraj Tiwari (abhirajtiwari@gmail.com)
    Sahil Khose (sahilkhose18@gmail.com)
"""
import args

import matplotlib.pyplot as plt
import os
import pandas as pd
import torch

from json import dumps
# print("__"*80)
# print("Imports Done... \n")
print("__"*80)

#TODO fetch saved generated images during training and their corresponding annotations
# annot = open(os.path.join(self.txt_dir, data_id + ".txt")).read().split("\n")[:-1]

def save_rgb_img():
	None

def write_log():
	None

def check_args():
    """
    To test args.py
    """
    print("get_data_args:")
    data_args = args.get_all_args()
    print(f'Args: {dumps(vars(data_args), indent=4, sort_keys=True)}')
    print("__"*80)

    print("To fetch arguments:")
    print(data_args.device)
    print(data_args.images_dir)
    print("__"*80)

    #####################################################################################
    print("To fetch images, text embeddings, bert embeddings:")
    df_train_filenames = pd.read_pickle(data_args.train_filenames)
    # print(df_train_filenames[0])  # List[str] : len train -> 8855, len test -> 2933
    image_path = os.path.join(data_args.images_dir, df_train_filenames[0] + ".jpg")
    text_path = os.path.join(data_args.annotations_dir, df_train_filenames[0] + ".txt")
    bert_path = os.path.join(data_args.bert_annotations_dir, df_train_filenames[0], "0.pt")


    print("\nBird type: ")
    print(df_train_filenames[0].split("/")[0])

    print("\nAnnotations of the bird image: \n")
    [print(f"{idx}: {ele}") for idx, ele in enumerate(open(text_path).read().split("\n")[:-1])]

    print("\nShape of bert embedding of annotation no 0:")
    emb = torch.load(bert_path)
    print(emb.shape)  # (1, 768)

    img = plt.imread(image_path)
    plt.imshow(img)
    plt.show()




if __name__ == "__main__":
    check_args()
