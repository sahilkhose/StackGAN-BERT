"""Command-line arguments.

Authors:
    Abhiraj Tiwari (abhirajtiwari@gmail.com)
    Sahil Khose (sahilkhose18@gmail.com)
"""
import argparse


def get_train_args():
	""" train.py """
	return None


def get_parameters(parser):
    '''Get parameters and hyper parameter values'''

    parser = argparse.ArgumentParser("Parameter args")
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="Device type: cuda/cpu")

    args = parser.parse_args()
    return args


def get_model_params():
    """
    Refer to StackGAN paper: https://arxiv.org/pdf/1612.03242.pdf 
    for parameter names.
    """
    parser = argparse.ArgumentParser("Model parameters")
    parser.add_argument("--n_g",
                        type=int,
                        default=128,
                        help="")
    parser.add_argument("--n_z",
                        type=int,
                        default=100,
                        help="")
    parser.add_argument("--m_d",
                        type=int,
                        default=4,
                        help="")
    parser.add_argument("--m_g",
                        type=int,
                        default=16,
                        help="")
    parser.add_argument("--n_d",
                        type=int,
                        default=128,
                        help="")
    parser.add_argument("--w_0",
                        type=int,
                        default=64,
                        help="")
    parser.add_argument("--h_0",
                        type=int,
                        default=256,
                        help="")
    parser.add_argument("--w",
                        type=int,
                        default=256,
                        help="")
    parser.add_argument("--h",
                        type=int,
                        default=256,
                        help="")

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
                        help="Annotations BERT embeddings dir path")
    parser.add_argument("--bert_path",
                        type=str,
                        default="../input/data/bert_base_uncased/",
                        help="Bert model dir path")
    parser.add_argument("--images_dir",
                        type=str,
                        default="../input/data/CUB_200_2011/images/",
                        help="Images dir path")

    ###* Files: 
    add_birds_file_args(parser)
    add_cub_file_args(parser)

    args = parser.parse_args()
    return args


def add_birds_file_args(parser):
    """
    Paths for files under input/data/birds
    files:
        - filenames.pickle            
            List[str] {filename DOES NOT contain .jpg (or any) extenstion}
            To fetch the image:           input/data/CUB_200_2011/images/<str>.jpg   == <images_dir>/<str>.jpg
            To fetch the text annotation: input/data/birds/text_c10/<str>.txt        == <annotations_dir>/<str>.txt
            To fetch the bert annotation: input/data/birds/embeddings/<str>/[0-9].pt == <bert_annotations_dir>/<str>/[0-9].pt
    There are 2 such files:
        input/data/birds/train/filenames.pickle : len = 8855 
        input/data/birds/test/filenames.pickle  : len = 2933
    """
    parser.add_argument("--train_filenames",
                        type=str,
                        default="../input/data/birds/train/filenames.pickle",
                        help="Pickle file path: filenames for train set")
    parser.add_argument("--test_filenames",
                        type=str,
                        default="../input/data/birds/test/filenames.pickle",
                        help="Pickle file path: filenames for test set")


def add_cub_file_args(parser):
    """
    Paths for files under input/data/CUB_200_2011
    files:
        - images.txt            
            <image_id> <image_name> {image_name contains .jpg extenstion}
            To fetch the image: input/data/CUB_200_2011/images/<image_name>
            number_of_images = 11788

        - train_test_split.txt
            <image_id> <is_training_image>
            1: train, 0: test
            train size = 5994
            test size = 5794

        - bounding_boxes.txt
            <image_id> <x> <y> <width> <height>
    """
    parser.add_argument("--images_id_file",
                        type=str,
                        default="../input/data/CUB_200_2011/images.txt",
                        help="Text file path: mapping image id to image path")  # <image_id> <image_name>

    #TODO re-run the bert embeddings code. Save the torch tensors using their ids.

    parser.add_argument("--train_test_split_file",
                        type=str,
                        default="../input/data/CUB_200_2011/train_test_split.txt",
                        help="Text file path: mapping image id to train/test split")  # <image_id> <is_training_image>
    
    parser.add_argument("--bounding_boxes",
                        type=str,
                        default="../input/data/CUB_200_2011/bounding_boxes.txt",
                        help="Text file path: mapping image id to train/test split")  # <image_id> <x> <y> <width> <height>
