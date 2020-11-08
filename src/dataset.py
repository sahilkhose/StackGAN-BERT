"""CUB Dataset class.

Authors:
    Abhiraj Tiwari (abhirajtiwari@gmail.com)
    Sahil Khose (sahilkhose18@gmail.com)
"""
import cv2
import numpy as np
import os
import pandas as pd
import torch


class CUBDataset(torch.utils.data.Dataset):
    def __init__(self, pickl_file, emb_dir, img_dir):
        self.file_names = pd.read_pickle(pickl_file)
        self.emb_dir = emb_dir
        self.img_dir = img_dir

    def __len__(self):
        # Total number of samples
        return len(self.file_names)

    def __getitem__(self, index):
        # Select sample
        data_id = str(self.file_names[index])

        # FIXME how to take text embedding input as each directory has 10 files
        text_emb = torch.load(os.path.join(self.emb_dir, data_id)+"/0.pt", map_location="cpu")
        image = torch.Tensor(cv2.imread(os.path.join(self.img_dir, data_id) + ".jpg"))

        return text_emb, image


if __name__ == "__main__":
    train_filenames = "../input/data/birds/train/filenames.pickle"
    test_filenames = "../input/data/birds/test/filenames.pickle"
    dataset_test = CUBDataset(train_filenames, "../input/data/birds/embeddings", "../input/data/CUB_200_2011/images")
    t, i = dataset_test[1]
    print(t.size(), i.size())
    