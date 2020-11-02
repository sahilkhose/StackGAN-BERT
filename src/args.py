"""Command-line arguments.

Authors:
    Abhiraj Tiwari (abhirajtiwari@gmail.com)
    Sahil Khose (sahilkhose18@gmail.com)
"""
import argparse

#TODO can add functions for different scripts (train, test, setup) like squad

def get_parameters():
    '''Get parameters and hyper parameter values'''

    parser = argparse.ArgumentParser("Parameter args")
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="Device type: cuda/cpu")

    args = parser.parse_args()
    return args



def get_data_args():
    '''Get all data paths'''

    parser = argparse.ArgumentParser("Data path args")
    ###* Directories:
    parser.add_argument("--annotations_dir",
                        type=str,
                        default="../input/data/birds/text_c10/",
                        help="Annotations dir path")
    parser.add_argument("--bert_annotations_dir",
                        type=str,
                        default="../input/data/birds/embeddings/",
                        help="Bert embeddings dir path for annotations")
    parser.add_argument("--bert_path",
                        type=str,
                        default="../input/data/bert_base_uncased/",
                        help="Bert model dir path")
    parser.add_argument("--images_dir",
                        type=str,
                        default="../input/data/CUB_200_2011/images/",
                        help="Images dir path")


    ###* Files: 
    parser.add_argument("--class_id_file",
                        type=str,
                        default="../input/data/CUB_200_2011/classes.txt",
                        help="Text file path: mapping class id to class name")  # <class_id> <class_name>
    parser.add_argument("--images_id_file",
                        type=str,
                        default="../input/data/CUB_200_2011/images.txt",
                        help="Text file path: mapping image id to image path")  # <image_id> <image_name>

    #TODO re-run the bert embeddings code. Save the torch tensors using their ids.

    parser.add_argument("--images_to_class_id_file",
                        type=str,
                        default="../input/data/CUB_200_2011/image_class_labels.txt",
                        help="Text file path: mapping image id to class id")  # <image_id> <class_id>
    parser.add_argument("--train_test_split_file",
                        type=str,
                        default="../input/data/CUB_200_2011/train_test_split.txt",
                        help="Text file path: mapping image id to train/test split")  # <image_id> <is_training_image>

    args = parser.parse_args()
    return args
