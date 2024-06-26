"""
ftocr_vl generated by microsoft OCR 0.05
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
import os
from PIL import Image
import imageio
import copy

from .augments import load_aug


class FTOCRDataset(Dataset):
    def __init__(
        self,
        mount_root: str = "/home/yilinjia/mycontainer/",
        source_ralative_path: str = "tengchao/dataset/kosmos_d/ft_ocr/sroie_funsd_cord_ft.json",
        data_dir_relative_path: str = "tengchao/dataset/kosmos_d/ft_ocr/", 
    ):
        source_path = os.path.join(mount_root, source_ralative_path)
        assert os.path.exists(
            source_path
        ), f"ftocr source path {source_path} not exists"

        self.data_dir = os.path.join(mount_root, data_dir_relative_path)
        assert os.path.exists(
            self.data_dir
        ), f"ftocr data dir {self.data_dir} not exists"

        # ftocr data
        self.source = json.load(open(source_path, "r"))
        assert len(self.source)==3, "ftocr source should have 3 parts"
        self.weight = [0.33, 0.33, 0.34]
        self.length = [len(i["source"])-1 for i in self.source]
        # Augmentation
        self.augment = load_aug()

        # define the whole dataset as an iterator
        # self.inf_iter = self.setup_iterator()
        self.inf_iter = self.base_generator()

    def __len__(self):
        return 100000000

    def __getitem__(self, index: int):
        return next(self.inf_iter)

    def concat(self, a, b):
        """
        {
            "img": pil_img,
            "lines": lines,
            "bboxs": bboxs,
        }
        """
        # first make the width the same
        width = max(a["img"].size[0], b["img"].size[0])
        a_height = int(width / a["img"].size[0] * a["img"].size[1])
        b_height = int(width / b["img"].size[0] * b["img"].size[1])
        height = a_height + b_height

        # resize and concat the image
        a["img"] = a["img"].resize((width, a_height))
        b["img"] = b["img"].resize((width, b_height))
        new_img = Image.new("RGB", (width, height))
        new_img.paste(a["img"], (0, 0))
        new_img.paste(b["img"], (0, a["img"].size[1]))
        new_lines = a["lines"] + b["lines"]
        new_bboxs = []
        for i in a["bboxs"]:
            x0, y0, x1, y1 = i
            # x0, x1 doesn't changes
            # y0 and y1 * a["img"].size[1] / height
            new_bboxs.append([x0, y0 * a["img"].size[1] / height, x1, y1 * a["img"].size[1] / height])
        for i in b["bboxs"]:
            x0, y0, x1, y1 = i
            # x0, x1 doesn't changes
            # y0 and y1 * b["img"].size[1] / height
            new_bboxs.append([x0, y0 * b["img"].size[1] / height + a["img"].size[1] / height, x1, y1 * b["img"].size[1] / height + a["img"].size[1] / height])
        return {
            "img": new_img,
            "lines": new_lines,
            "bboxs": new_bboxs,
        }

    def base_generator(self):
        def single_generator():
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
        return single_generator()
    
    def setup_iterator(self):
        base_gen = self.base_generator()
        def inf_shuffle_generator():
            while True:
                a = next(base_gen)
                if random.random() < 0.5:
                    yield a
                else:
                    b = next(base_gen)
                    yield self.concat(a, b)
        return inf_shuffle_generator()

    def read_file(self, file_relative_path: str):
        """
        This is a generator

        input:
            sroie/images/X00016469612.jpg
            sroie/ocrs/X00016469612.txt

        output:
        {
            img: PIL.Image
            lines: List[str]
            bboxs: List[List[float]]
        }

        """
        try:
            file_path = os.path.join(self.data_dir, file_relative_path)
            if not os.path.exists(file_path):
                print("| file {} not exists".format(file_path), flush=True)
                return iter([])

            # original image
            image = imageio.imread(f"{file_path}", pilmode="RGB")

            # Augmentation 90% of the time
            aug_and_mask_flag = random.randint(0, 9) != 0
            if aug_and_mask_flag:
                cur_augs = random.sample(self.augment, 1)
                for aug_list in cur_augs:
                    cur_aug = random.choice(aug_list)
                    image = cur_aug(image=image)
            pil_img = Image.fromarray(image)

            original_width, original_height = pil_img.size

            # assert original_width * 8 >= original_height, "img too high"
            # assert original_height * 8 >= original_width, "img too wide"

            if "sroie" in file_path:
                lines = []
                bboxs = []
                info_root = file_path.replace("images", "ocrs").replace("jpg", "txt")
                alllines = open(f"{info_root}", encoding="utf-8").readlines()
                result = ""
                for line in alllines:
                    items = line.strip().split(",")
                    x0, y0, x1, y1, x2, y2, x3, y3 = items[:8]
                    if len(items) > 9:
                        text = ",".join(items[8:]).lower()
                    else:
                        text = items[8].lower()

                    xmin = min(int(x0), int(x1), int(x2), int(x3))
                    ymin = min(int(y0), int(y1), int(y2), int(y3))
                    xmax = max(int(x0), int(x1), int(x2), int(x3))
                    ymax = max(int(y0), int(y1), int(y2), int(y3))
                    result += f"{text}\n"
                    lines.append(text)
                    bboxs.append(
                        [
                            xmin / original_width,
                            ymin / original_height,
                            xmax / original_width,
                            ymax / original_height,
                        ]
                    )
            elif "funsd" in file_path:
                lines = []
                bboxs = []
                info_root = file_path.replace("images", "ocrs").replace("png", "txt")
                alllines = open(f"{info_root}", encoding="utf-8").readlines()
                result = ""
                for line in alllines:
                    items = line.strip().split(",")
                    x0, y0, x1, y1, x2, y2, x3, y3 = items[:8]
                    if len(items) > 9:
                        text = ",".join(items[8:])
                    else:
                        text = items[8]

                    xmin = min(int(x0), int(x1), int(x2), int(x3))
                    ymin = min(int(y0), int(y1), int(y2), int(y3))
                    xmax = max(int(x0), int(x1), int(x2), int(x3))
                    ymax = max(int(y0), int(y1), int(y2), int(y3))
                    result += f"{text}\n"
                    lines.append(text)
                    bboxs.append(
                        [
                            xmin / original_width,
                            ymin / original_height,
                            xmax / original_width,
                            ymax / original_height,
                        ]
                    )
            elif "cord" in file_path:
                lines = []
                bboxs = []
                info_root = file_path.replace("images", "ocrs").replace("png", "txt")
                alllines = open(f"{info_root}", encoding="utf-8").readlines()
                result = ""
                for line in alllines:
                    items = line.strip().split(",")
                    x0, y0, x1, y1, x2, y2, x3, y3 = items[:8]
                    if len(items) > 9:
                        text = ",".join(items[8:]).lower()
                    else:
                        text = items[8].lower()

                    xmin = min(int(x0), int(x1), int(x2), int(x3))
                    ymin = min(int(y0), int(y1), int(y2), int(y3))
                    xmax = max(int(x0), int(x1), int(x2), int(x3))
                    ymax = max(int(y0), int(y1), int(y2), int(y3))
                    result += f"{text}\n"
                    lines.append(text)
                    bboxs.append(
                        [
                            xmin / original_width,
                            ymin / original_height,
                            xmax / original_width,
                            ymax / original_height,
                        ]
                    )
            else:
                # md is not ocr task
                return iter([])
                lines = []
                bboxs = []
                info_root = file_path.replace("images", "mds").replace("jpg", "md")
                line = open(f"{info_root}").read()
                lines = [line]
                bboxs = [[0, 0, 0, 0]]
            yield {
                "img": pil_img,
                "lines": lines,
                "bboxs": bboxs,
            }
        except:
            return iter([])

        # finish reading all the entries in the file
        return iter([])
