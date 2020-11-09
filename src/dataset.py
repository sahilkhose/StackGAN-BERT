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

    ## Bounding box text file: len: 11788
    ## <id> <x> <y> <width> <height>
    bbox_file = open(data_args.bounding_boxes).read().split("\n")[:-1]
    idx = 0
    print(f"{bbox_file[idx].split()[0]} ({bbox_file[idx].split()[1]}, {bbox_file[idx].split()[2]}, {bbox_file[idx].split()[3]}, {bbox_file[idx].split()[4]})")

    list_ids = [i+1 for i in range(len(bbox_file))]
    list_coords = [(float(bbox_file[idx_].split()[1]), float(bbox_file[idx_].split()[2]), float(bbox_file[idx_].split()[3]), float(bbox_file[idx_].split()[4])) for idx_ in range(len(list_ids))]
    print(list_ids[idx], list_coords[idx])

    ## image id file:
    ## <id> <filename.jpg>
    image_id_file = open(data_args.images_id_file).read().split("\n")[:-1]
    print(f'{image_id_file[idx].split()[0]} {image_id_file[idx].split()[-1].replace(".jpg", "")}')
    list_fnames = [image_id_file[idx_].split()[-1].replace(".jpg", "") for idx_ in range(len(list_ids))]
    print(list_ids[idx], list_fnames[idx])

    print("image id of the filename is: ", list_ids[list_fnames.index("001.Black_footed_Albatross/Black_Footed_Albatross_0046_18")])
    print("bbox coords of filename are: ", list_coords[list_fnames.index("001.Black_footed_Albatross/Black_Footed_Albatross_0046_18")])