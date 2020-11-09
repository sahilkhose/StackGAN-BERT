"""CUB Dataset class.

Authors:
    Abhiraj Tiwari (abhirajtiwari@gmail.com)
    Sahil Khose (sahilkhose18@gmail.com)
"""
import args

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
print("__"*80)


class CUBDataset(torch.utils.data.Dataset):
    def __init__(self, pickl_file, emb_dir, img_dir, txt_dir):
        self.file_names = pd.read_pickle(pickl_file)
        self.emb_dir = emb_dir
        self.img_dir = img_dir
        self.txt_dir = txt_dir

    def __len__(self):
        # Total number of samples
        return len(self.file_names)

    def __getitem__(self, index):
        # Select sample
        data_id = str(self.file_names[index])
        print(data_id)

        # FIXME how to take text embedding input as each directory has 10 files
        text_emb = torch.load(os.path.join(self.emb_dir, data_id)+"/0.pt", map_location="cpu")
        # image = torch.Tensor(cv2.imread(os.path.join(self.img_dir, data_id) + ".jpg"))
        image = cv2.imread(os.path.join(self.img_dir, data_id) + ".jpg")
        annot = open(os.path.join(self.txt_dir, data_id + ".txt")).read().split("\n")[:-1]

        return text_emb, image, annot


if __name__ == "__main__":
    data_args = args.get_data_args()
    train_filenames = data_args.train_filenames
    test_filenames = data_args.test_filenames
    dataset_test = CUBDataset(train_filenames, data_args.bert_annotations_dir,
                              data_args.images_dir, data_args.annotations_dir)
    t, i, a = dataset_test[1]
    print("Bert emb shape: ", t.shape)
    print("Image shape: ", i.shape)
    print("Annotations:")
    [print(f"{idx}: {ele}") for idx, ele in enumerate(a)]
    
    plt.imshow(i)
    plt.show()
    
