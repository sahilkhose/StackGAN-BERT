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
    def __init__(self, pickl_file, emb_dir, img_dir, bbox_dir):
        self.file_names = pd.read_pickle(pickl_file)
        self.emb_dir = emb_dir
        self.img_dir = img_dir
        self.bbox_dir = bbox_dir

    def __len__(self):
        # Total number of samples
        return len(self.file_names)

    def __getitem__(self, index):
        # Select sample
        data_id = str(self.file_names[index])
        print(data_id)
        image = cv2.imread(os.path.join(self.img_dir, data_id) + ".jpg")
        
        # FIXME how to take text embedding input as each directory has 10 files
        text_emb = torch.load(os.path.join(self.emb_dir, data_id)+"/0.pt", map_location="cpu")
        image = torch.Tensor(cv2.imread(os.path.join(self.img_dir, data_id) + ".jpg"))
        bbox 
        return text_emb, image


def dict_bbox():
    """
    returns filename to bbox dict
    """
    data_args = args.get_data_args()

    df_bbox = pd.read_csv(data_args.bounding_boxes, delim_whitespace=True, header=None).astype(int)
    df_filenames = pd.read_csv(data_args.images_id_file, delim_whitespace=True, header=None)

    filenames = df_filenames[1].tolist()

    filename_bbox = {}
    for i in range(len(filenames)):
        bbox = df_bbox.iloc[i][1:].tolist()
        key = filenames[i].replace(".jpg", "")
        filename_bbox[key] = bbox
    
    return filename_bbox


if __name__ == "__main__":
    data_args = args.get_data_args()
    # train_filenames = data_args.train_filenames
    # test_filenames = data_args.test_filenames
    # dataset_test = CUBDataset(train_filenames, data_args.bert_annotations_dir, data_args.images_dir)
    # t, i = dataset_test[1]
    # print("Bert emb shape: ", t.shape)
    # print("Image shape: ", i.shape)
    
    # plt.imshow(i)
    # plt.show()
    

    ###########################################################
    filename = "001.Black_footed_Albatross/Black_Footed_Albatross_0046_18"
    f_to_bbox = dict_bbox()
    print(f_to_bbox[filename])