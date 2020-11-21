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
import torchvision.utils as vutils

from json import dumps
print("__"*80)

#TODO fetch saved generated images during training and their corresponding annotations

def save_img_results(data_img, fake, epoch, args):
    num = args.VIS_COUNT
    fake = fake[0:num]
    # data_img is changed to [0, 1]
    if data_img is not None:
        data_img = data_img[0:num]
        vutils.save_image(data_img, os.path.join(args.image_save_dir, "real_samples.png"), normalize=True)
        # fake data is stil [-1, 1]
        vutils.save_image(fake.data, os.path.join(args.image_save_dir, f"fake_samples_epoch_{epoch:04}.png"), normalize=True)
    else:
        vutils.save_image(fake.data, os.path.join(args.image_save_dir, f"lr_fake_samples_epoch_{epoch:04}.png"), normalize=True)

def save_model(netG, netD, epoch, args):
    torch.save(netG.state_dict(), os.path.join(args.model_dir, f"netG_epoch_{epoch}.pth"))
    torch.save(netD.state_dict(), os.path.join(args.model_dir, f"netD_epoch_{epoch}.pth")) #? implementation has saved just the last disc? decision?
    print("Save G/D models")


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_dataset(training_set):
    t, i, b = training_set[1]
    print("Bert emb shape: ", t.shape)
    print("bbox: ", b)
    plt.imshow(i)
    plt.show()
    print("__"*80)

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
    # make_dir("../output/model")
