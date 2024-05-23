"""
Handwritten font dataset
    generate img with different fonts
output:
{
    img: PIL.Image
    lines: List[str]
    bboxs: List[List[float]]
}
"""

from torch.utils.data import Dataset
from typing import List
import random
import json
from .generator import build_text_img
from argparse import Namespace
import os
import io
from PIL import Image
import imageio
import base64
import glob

# from augments import load_aug


class HWFontDataset(Dataset):
    def __init__(
        self,
        mount_root: str = "/home/yilinjia/mycontainer/",
        font_relative_path: str = "tengchao/dataset/kosmos_d/handwritten_fonts/",
        text_source_relative_path: str = "shaohanh/data/tnlg_config/json/train.json",
        text_data_dir: str = "shaohanh/data/tnlg/",
    ):
        font_path = os.path.join(mount_root, font_relative_path)
        assert os.path.exists(font_path), "HW fonts doesn't exist"

        text_source_path = os.path.join(mount_root, text_source_relative_path)
        assert os.path.exists(text_source_path), "HW text source doesn't exist"

        # Different type of fonts
        self.fonts = glob.glob(os.path.join(font_path, "*.ttf"))
        # Text source
        self.source = json.load(open(text_source_path, "r"))
        self.weight = [i['weight'] for i in self.source]
        self.length = [len(i['source']) - 1 for i in self.source]
        # Text data directory
        self.data_dir = os.path.join(mount_root, text_data_dir)
        assert os.path.exists(self.data_dir), "HW text data directory doesn't exist"
        # Augmentation
        # self.augment = load_aug()
        # Infinite dataset
        self.inf_iter = self.setup_iterator()

    def __len__(self):
        return 100000000

    def setup_iterator(self):
        # Create an infinite iterator
        def inf_shuffle_generator():
            while True:
                i = random.choices(range(len(self.source)), weights=self.weight)[0]
                entry = self.source[i]
                j = random.randint(0, self.length[i])
                source_file = entry["source"][j]
                iterator = self.read_file(source_file)
                try:
                    while True:
                        yield next(iterator)
                except StopIteration:
                    continue     
        return inf_shuffle_generator()
        
    def __getitem__(self, i):
        return next(self.inf_iter)

    def read_file(self, txt_file: str):
        """
        read the file and reture a a generator
        output:
            {
                "lines": text_extract, 
                "bboxs": bboxs, 
                "img": pix
            }
        """
        # check whether the txt file exist
        file_path = os.path.join(self.data_dir, txt_file)
        if not os.path.exists(file_path):
            print("| file {} not exists".format(file_path), flush=True)
            return iter([])

        # read the entries from the source file
        try:
            with open(file_path, "r", encoding="utf8") as f:
                entries = f.read().strip().split("\n")
        except:
            return iter([])

        # construct the iterator
        try:
            # shuffle the entries
            random.shuffle(entries)
            for et in entries:
                try:
                    entry = json.loads(et)
                    yield build_text_img(
                        entry["text"],
                        random_text=False,
                        cur_font=random.choice(self.fonts),
                    )
                except Exception as e:
                    continue
        except Exception as e:
            return iter([])

        # finish reading all the entries in the file
        return iter([])
