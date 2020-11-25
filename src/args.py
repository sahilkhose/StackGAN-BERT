"""Command-line arguments.

Authors:
    Abhiraj Tiwari (abhirajtiwari@gmail.com)
    Sahil Khose (sahilkhose18@gmail.com)
"""
# >  python3 args.py --conf ../cfg/s1.yml
import argparse
import yaml
from json import dumps

########################################### MAIN FUNCTIONS ###########################################

def get_all_args():
    return parse_args(get_parser())


def print_args(args):
    print("__"*80)
    print(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    print("__"*80)


def parse_args(parser):
    args = parser.parse_args()
    if args.config_file:
        data = yaml.load(args.config_file, Loader=yaml.FullLoader)
        delattr(args, "config_file")
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    return args


def get_parser():
    """Get all parameters"""
    parser = argparse.ArgumentParser("Arguments")

    add_conf_args(parser)
    add_train_args(parser)
    add_model_args(parser)
    add_data_args(parser)

    return parser


########################################### PARSER ARGUMENTS ###########################################
def add_conf_args(parser):
    parser.add_argument("--config-file",
                        type=argparse.FileType(mode="r"),
                        dest="config_file",
                        help="config yaml file to pass params")


def add_train_args(parser):
    """ train.py """
    parser.add_argument("--NET_G_path",
                        type=str,
                        default="",
                        help="Generator model loading path")
    parser.add_argument("--NET_D_path",
                        type=str,
                        default="",
                        help="Discriminator model loading path")
    parser.add_argument("--STAGE1_G_path",
                        type=str,
                        default="",
                        help="Stage 1 Generator model path for Stage 2 training")
    parser.add_argument("--train_bs",
                        type=int,
                        default=2,
                        help="train batch size")
    parser.add_argument("--test_bs",
                        type=int,
                        default=2,
                        help="test batch size")
    parser.add_argument("--train_workers",
                        type=int,
                        default=1,
                        help="train num_workers")
    parser.add_argument("--test_workers",
                        type=int,
                        default=1,
                        help="test num_workers")
    parser.add_argument("--TRAIN_GEN_LR",
                        type=float,
                        default=2e-4,
                        help="train generator learning rate")
    parser.add_argument("--TRAIN_DISC_LR",
                        type=float,
                        default=2e-4,
                        help="test discriminator learning rate")
    parser.add_argument("--TRAIN_LR_DECAY_EPOCH",
                        type=int,
                        default=600,
                        help="train lr decay epoch")
    parser.add_argument("--TRAIN_MAX_EPOCH",
                        type=int,
                        default=600,
                        help="train maximum epochs")
    parser.add_argument("--TRAIN_SNAPSHOT_INTERVAL",
                        type=int,
                        default=50,
                        help="Snapshot interval")
    parser.add_argument("--TRAIN_COEFF_KL",
                        type=float,
                        default=2.0,
                        help="train coefficient KL")
    parser.add_argument("--dataset_name",
                        type=str,
                        default="birds",
                        help="birds/flowers: dataset name")
    parser.add_argument("--embedding_type",
                        type=str,
                        default="bert",
                        help="bert/cnn-rnn: embedding type")
    parser.add_argument("--datapath",
                        type=str,
                        default="../output",
                        help="datapath dir")
    parser.add_argument("--image_save_dir",
                        type=str,
                        default="../output/image/",
                        help="Image save dir")
    parser.add_argument("--model_dir",
                        type=str,
                        default="../output/model/",
                        help="Model save dir")
    parser.add_argument("--log_dir",
                        type=str,
                        default="../output/log/",
                        help="Log dir for tensorboard")
    parser.add_argument("--VIS_COUNT",
                        type=int,
                        default=64,
                        help="")
    parser.add_argument("--STAGE",
                        type=int,
                        default=1,
                        help="Stage to train/eval (1/2)")
    parser.add_argument("--device",
                        type=str,
                        default="cuda", #! CHANGE THIS TO CUDA BEFORE TRAINING
                        help="Device type: cuda/cpu")


def add_model_args(parser):
    """
    Refer to StackGAN paper: https://arxiv.org/pdf/1612.03242.pdf 
    for parameter names.
    """
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


def add_data_args(parser):
    '''Get all data paths'''
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
    parser.add_argument("--cnn_annotations_emb_train",
                        type=str,
                        default="../input/data/birds/train/char-CNN-RNN-embeddings.pickle",
                        help="char-CNN-RNN-embeddings pickle file for train")
    parser.add_argument("--cnn_annotations_emb_test",
                        type=str,
                        default="../input/data/birds/test/char-CNN-RNN-embeddings.pickle",
                        help="char-CNN-RNN-embeddings pickle file for test")
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
    parser.add_argument("--train_test_split_file",
                        type=str,
                        default="../input/data/CUB_200_2011/train_test_split.txt",
                        help="Text file path: mapping image id to train/test split")  # <image_id> <is_training_image>
    parser.add_argument("--bounding_boxes",
                        type=str,
                        default="../input/data/CUB_200_2011/bounding_boxes.txt",
                        help="Text file path: mapping image id to train/test split")  # <image_id> <x> <y> <width> <height>


if __name__ == "__main__":
    args = get_all_args()
    print_args(args)
    print("args:", args.train_bs)
