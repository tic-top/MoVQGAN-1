"""
Contains the dataset classes for the OCR model.

Mixed dataset:
    cdip_ocr_vl pdf/cdip 0.855
    hwfont_ocr handwritien simple image 0.0475
    gpt_ocr synthetic simple image 0.0475
    ftocr_vl generated with microsoft OCR 0.05

output:
{
    img: PIL.Image
    lines: List[str]
    bboxs: List[List[float]]
}
"""

import os
from torch.utils.data import Dataset
import random
import json
import fasttext
import io
import imageio
import base64
from PIL import Image, ImageDraw
from .augments import load_aug

LA_THRESHOLD = 0.7


def lid(text, model):
    span_start, span_end = 0, len(text)
    det_text = text[span_start:span_end]
    res = model.predict(det_text)
    la = res[0][0].replace("__label__", "")
    prob = res[1][0]
    return la, prob


class CDIPDataset(Dataset):
    def __init__(
        self,
        mount_root: str = "/home/yilinjia/mycontainer/",
        source_relative_path: str = "tengchao/dataset/kosmos_d/ocr_line/train.json",
        data_dir_relative_path: str = "tengchao/dataset/kosmos_d/ocr_line/",  # and /parallel
    ):
        source_path = os.path.join(mount_root, source_relative_path)
        assert os.path.exists(source_path), f"cdip source path {source_path} not exists"

        self.data_dir = os.path.join(mount_root, data_dir_relative_path)
        # self.receipt_dir = os.path.join(mount_root, "dataset")
        assert os.path.exists(
            self.data_dir
        ), f"cdip data dir {self.data_dir} not exists"

        # Source files
        self.source = json.load(open(source_path, "r"))
        self.weight = [i["weight"] for i in self.source]
        # make the weight of "receipts" to be 0 this dataset is dirty
        # self.weight = [0.5 if i == "receipts" else i for i in self.weight]
        self.length = [len(i["source"]) - 1 for i in self.source]
        # Augmentation
        self.augment = load_aug()

        # Language Identifier
        self.lid_model = fasttext.load_model(
            os.path.join(mount_root, "tengchao/dataset/models/lid.176.bin")
        )

        # define the whole dataset as an iterator
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

    def mask_img(self, img, bboxs, width, height, mask_ratio):
        def paint(image, bbox, width, height):
            def clip(min_num, num, max_num):
                return min(max(num, min_num), max_num)

            x0, y0, x1, y1 = bbox

            x0 = clip(0, int(x0 * width), width)
            y0 = clip(0, int(y0 * height), height)
            x1 = clip(0, int(x1 * width), width)
            y1 = clip(0, int(y1 * height), height)

            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    image.putpixel((x, y), (0, 0, 0))

            draw = ImageDraw.Draw(image)
            draw.line([(x0, y0), (x1, y1)], fill="red", width=5)
            draw.line([(x0, y1), (x1, y0)], fill="red", width=5)
            draw.line([(x0, y0), (x0, y1)], fill="red", width=5)
            draw.line([(x0, y1), (x1, y1)], fill="red", width=5)
            draw.line([(x1, y1), (x1, y0)], fill="red", width=5)
            draw.line([(x1, y0), (x0, y0)], fill="red", width=5)

            return image

        if len(bboxs) < 3:
            return img

        mask_num = int(len(bboxs) * mask_ratio)
        mask_num = max(min(int(len(bboxs) / 2), mask_num), 0)

        bbox_masked = random.sample(bboxs, mask_num)
        for bbox in bbox_masked:
            img = paint(img, bbox, width, height)
        return img

    def read_file(self, file_relative_path: str):
        """
        This is a generator
        """
        try: 
            file_path = os.path.join(self.data_dir, file_relative_path)
            if "parallel" in file_relative_path:
                # replace ocr_line with parallel
                file_path = file_path.replace("ocr_line", "parallel")
            if not os.path.exists(file_path):
                print("| file {} not exists".format(file_path), flush=True)
                return iter([])
            # entries is a list of dict contains ['image', 'lines', 'bboxs']
            with open(file_path, "r", encoding="utf8") as f:
                entries = f.read().strip().split("\n")
            # shuffle the entries
            random.shuffle(entries)
            for entry_s in entries:
                # entry_s is a string, we need convert it to json
                try:
                    assert len(entry_s.strip()) != 0
                    entry = json.loads(entry_s.strip())
                    if "boxes" in entry:
                        entry["bboxs"] = entry["boxes"]
                        del entry["boxes"]
                    assert len(entry["bboxs"]) == len(
                        entry["lines"]
                    ), f"bboxs and lines not match: {entry['bboxs']} {entry['lines']}"

                    # validation
                    if "pdf" in file_relative_path:
                        la, prob = lid(" ".join(entry["lines"]), self.lid_model)
                        if la not in "en":
                            continue
                        if prob < LA_THRESHOLD:
                            continue
                    # validation
                    pic = Image.open(
                        io.BytesIO(base64.b64decode(entry["image"]))
                    ).convert("RGB")
                    extrema = pic.convert("L").getextrema()
                    assert extrema[0] != extrema[1], "image is blank"
                    del pic
                    pic = imageio.imread(
                        io.BytesIO(base64.b64decode(entry["image"])), pilmode="RGB"
                    )
                    original_width, original_height, _ = pic.shape
                    # print(original_width, original_height)
                    assert original_width * 8 >= original_height, "img too high"
                    assert original_height * 8 >= original_width, "img too wide"
                    # Augmentation 90% of the time
                    aug_and_mask_flag = random.randint(0, 9) != 0
                    aug_and_mask_flag = False
                    if aug_and_mask_flag:
                        cur_augs = random.sample(self.augment, 1)
                        for aug_list in cur_augs:
                            cur_aug = random.choice(aug_list)
                            pic = cur_aug(image=pic)
                    pic = Image.fromarray(pic)
                    if (
                        aug_and_mask_flag
                        and "handwritten" not in file_path
                        and "mt_sj" not in file_path
                        and "parallel_chrome_math" not in file_path
                    ):
                        pic = self.mask_img(
                            pic,
                            entry["bboxs"],
                            width=pic.width,
                            height=pic.height,
                            mask_ratio=0.3,
                        )
                    yield {
                        "img": pic,
                        # "lines": entry["lines"],
                        # "bboxs": entry["bboxs"],
                    }
                except Exception as e:
                    continue

        except Exception as e:
            return iter([])
        # finish reading all the entries in the file
        return iter([])
