# >  python3 args.py --conf birds_s1.yml
import argparse
import yaml
from json import dumps

def get_parser():
    """Get all parameters"""

    parser = argparse.ArgumentParser("Arguments")

    parser.add_argument("--config-file",
                        type=argparse.FileType(mode="r"),
                        dest="config_file",
                        help="config yaml file to pass params")
    parser.add_argument("--train.train_bs",
                        type=int,
                        default=1,
                        help="batchsize of train")
    parser.add_argument("--train.epochs",
                        type=int,
                        default=1,
                        help="epochs of train")
    return parser

def parse_args(parser):
    args = parser.parse_args()
    if args.config_file:
        data = yaml.load(args.config_file)
        delattr(args, "config_file")
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
        
    return args
        

def print_args(args):
    print("__"*80)
    print(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    print("__"*80)

def get_all_args():
    return parse_args(get_parser())


if __name__ == "__main__":
    args = get_all_args()
    print_args(args)
    print("args:", args.train["train_bs"])
