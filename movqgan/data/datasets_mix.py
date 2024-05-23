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
from typing import List
import random

class MixDataset(Dataset):
    """
    output:
            {
                "lines": text_extract,
                "bboxs": bboxs,
                "img": pix
            }
    """

    def __init__(
        self,
        datasets: List[Dataset],
        weights: List[float] = [0.85, 0.05, 0.05, 0.05],
        processor=None,
    ):
        self.datasets = datasets
        self.processor = processor
        self.weights = weights
        # check whether the length of datasets and weights are the same
        assert len(datasets) == len(weights)

    def __len__(self):
        return 100000000

    def __getitem__(self, i) -> str:
        # randomly select a dataset based on the weights
        dataset_idx = random.choices(range(len(self.datasets)), weights=self.weights)[0]
        if self.processor is None:
            return self.datasets[dataset_idx][i]
        return self.processor(self.datasets[dataset_idx][i])
        # getted = False
        # while not getted:
        #     try:
        #         x = self.datasets[dataset_idx][i]
        #         getted = True
        #     except:
        #         pass
        # return x
