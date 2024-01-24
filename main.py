import torch
import torch.nn as nn
import argparse
import os
from tqdm import tqdm
from ZS_NSN import Image_denoise



def get_parser():
    parser = argparse.ArgumentParser(description="Dual Deep Mesh Prior")
    parser.add_argument("-i", "--input", type=str, required=True)
    args = parser.parse_args()
    return args.input


def main():
    global image_path
    args = get_parser()
    image_path=get_parser()
    Image_denoise(image_path)


if __name__ == "__main__":
    main()