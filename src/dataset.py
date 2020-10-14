import config

import cv2
import numpy as np
import os
import pandas as pd
import torch


class CUBDataset:
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        # Total number of samples
        return len(self.ids)

    def __getitem__(self, index):
        # Select sample
        data_id = str(self.ids[index])

        # Load data and get label
        text_emb = torch.load(os.path.join(config.TEXT_EMB, data_id), map_location="cpu")
        image = cv2.imread(os.path.join(config.IMG, data_id + ".png"))

        return text_emb, image