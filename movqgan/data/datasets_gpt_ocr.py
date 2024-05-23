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
from torch.utils.data import Dataset
import random
import json
from .generator import build_text_img
import os

class GPTOCRDataset(Dataset):
    def __init__(
        self,
        mount_root: str = "/home/yilinjia/mycontainer/",
        text_source_relative_path: str = "shaohanh/data/tnlg_config/json/train.json",
        text_data_dir: str = "shaohanh/data/tnlg/",
        task: str = "warmup",
    ):
        self.random_align = task != "warmup"
        if self.random_align:
            print(f"random align {task}")
        text_source_path = os.path.join(mount_root, text_source_relative_path)
        assert os.path.exists(text_source_path), f"GPT text source doesn't exist, {text_source_path}"
        # Text source
        self.source = json.load(open(text_source_path, "r"))
        self.weight = [i['weight'] for i in self.source]
        self.length = [len(i['source']) - 1 for i in self.source]
        # Text data directory
        self.data_dir = os.path.join(mount_root, text_data_dir)
        assert os.path.exists(self.data_dir), "GPT text data directory doesn't exist"
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
        # check whether the source file exist
        file_path = os.path.join(self.data_dir, txt_file)
        if not os.path.exists(file_path):
            print('| file {} not exists'.format(file_path), flush=True)
            return iter([])
        
        # read the entries from the source file
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                entries = f.read().strip().split('\n')
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
                        entry['text'],
                        random_text=True,
                        align = random.randint(0, 2) if self.random_align else 0,
                    )
                except Exception as e:
                    continue
        except Exception as e:
            return iter([])
        
        # finish reading all the entries in the file
        return iter([])