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
import pickle
import PIL
import torch

from torchvision import transforms

from PIL import Image
print("__"*80)


class CUBDataset(torch.utils.data.Dataset):
    def __init__(self, pickl_file, img_dir, bert_emb=None, cnn_emb=None, stage=1):
        self.file_names = pd.read_pickle(pickl_file)[:-23]
        self.img_dir = img_dir
        self.bert_emb = bert_emb
        if cnn_emb is not None:
            self.cnn_emb = np.array(pickle.load(open(cnn_emb, "rb"), encoding='latin1'))  # to switch to cnn-rnn embeddings
        else:
            self.cnn_emb = cnn_emb
        self.stage = stage
        self.f_to_bbox = dict_bbox()

    def __len__(self):
        # Total number of samples
        return len(self.file_names)

    def __getitem__(self, index):
        # Select sample:
        data_id = str(self.file_names[index])

        # Fetch text emb, image, bbox:
        idx = np.random.randint(0, 9)
        if self.cnn_emb is not None:
            text_emb = torch.tensor(self.cnn_emb[index][idx], dtype=torch.float)  # (1024)
        else:
            text_emb = torch.load(os.path.join(self.bert_emb, data_id)+f"/{idx}.pt", map_location="cpu")
            text_emb = text_emb.squeeze(0)  # (768)
        
        bbox = self.f_to_bbox[data_id]
        if self.stage == 1:
            image = get_img(img_path=os.path.join(self.img_dir, data_id) + ".jpg", bbox=bbox, image_size=(64, 64))
        else:
            image = get_img(img_path=os.path.join(self.img_dir, data_id) + ".jpg", bbox=bbox, image_size=(256, 256))
        # image = torch.tensor(np.array(image), dtype=torch.float)
        image = transforms.ToTensor()(image)

        return text_emb, image


def dict_bbox():
    """
    returns filename to bbox dict
    """
    data_args = args.get_all_args()

    df_bbox = pd.read_csv(data_args.bounding_boxes, delim_whitespace=True, header=None).astype(int)
    df_filenames = pd.read_csv(data_args.images_id_file, delim_whitespace=True, header=None)

    filenames = df_filenames[1].tolist()

    filename_bbox = {}
    for i in range(len(filenames)):
        bbox = df_bbox.iloc[i][1:].tolist()
        key = filenames[i].replace(".jpg", "")
        filename_bbox[key] = bbox
    
    return filename_bbox

def get_img(img_path, bbox, image_size):
    """
    Load and resize image
    """
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - R)
        y2 = np.minimum(height, center_y + R)
        x1 = np.maximum(0, center_x - R)
        x2 = np.minimum(width, center_x + R)
        img = img.crop([x1, y1, x2, y2])
    img = img.resize(image_size, PIL.Image.BILINEAR)
    return img

if __name__ == "__main__":
    data_args = args.get_all_args()
    train_filenames = data_args.train_filenames
    test_filenames = data_args.test_filenames

    bert_embed = False

    if bert_embed:
        ###* Bert embeddings dataset:
        dataset_test = CUBDataset(train_filenames, data_args.images_dir, bert_emb=data_args.bert_annotations_dir)
    else:
        ###* cnn-rnn embeddings dataset:
        # cnn_embeddings = np.array(pickle.load(open(data_args.cnn_annotations_emb_train, "rb"), encoding='latin1'))
        dataset_test = CUBDataset(train_filenames, data_args.images_dir, cnn_emb=data_args.cnn_annotations_emb_train)


    t, i = dataset_test[1]
    if bert_embed:
        print("Bert emb shape: ", t.shape)
    else: 
        print("rnn-cnn emb shape: ", t.shape)
    print("Image shape: ", i.shape)
    plt.imshow(i.permute(1, 2, 0))
    plt.show()

    ###########################################################
    # filename = "001.Black_footed_Albatross/Black_Footed_Albatross_0046_18"
    # f_to_bbox = dict_bbox()
    # print(f_to_bbox[filename])
