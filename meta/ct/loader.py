"""
Author: Ibrahim Almakky
Date: 03/02/2022
"""
import os
from glob import glob
import json
import torch
import torch.utils.data


DATASETS = [
    "kits",
    "fetal_ultrasound",
    "MosMed",
    "LiTs",
    "RSPECT",
    "IHD_Brain",
    "ImageCHD",
    "CTPancreas",
    "Brain_MRI",
    "ProstateMRI",
    "RSNAXRay",
    "Covid19XRay"
]

MODES = ["Classify", "Segment"]
SLICE_EXT = ".pt"
SLICES_DIR = "slices"

CLASSES = {
    "Segment": {
        "MosMed": {"Pos": [1]},
        "kits": {"Benign": [1], "Malignant": [2]},
        "LiTs": {"Tumor": [1]},
        "ImageCHD": {
            "Heart": [1, 2, 3, 4, 5, 6, 7],
            "CHD": [8, 9, 10, 11, 12, 13, 14, 15],
        },
        "CTPancreas": [],
    },
    "Classify": {
        "MosMed": ["Neg", "Pos"],
        "kits": ["Benign", "Malignant"],
        "LiTs": ["No_Tumor", "Tumor"],
        "RSPECT": ["No_PE", "PE"],
        "IHD_Brain": ["No_IHD", "IHD"],
        "IHD_Brain_Multi": [
            "No_IHD",
            "epidural",
            "intraparenchymal",
            "intraventricular",
            "subarachnoid",
            "subdural",
        ],
        "ImageCHD": ["No_CHD", "CHD"],
        "CTPancreas": ["No_Tumor", "Tumor"],
        "Brain_MRI": [
            "glioma_tumor",
            "meningioma_tumor",
            "no_tumor",
            "pituitary_tumor",
        ],
    },
}

DATASET_ATTR = {
    "MosMed": {
        "organ": "Lungs",
        "pathologies": ["COVID-19"],
        "location": ["Russia"],
        "size": 3200,
        "modality": "CT",
        "classes": ["COVID-19 Positive", "COVID-19 Negative"],
    },
    "kits": {
        "organ": "Kidney",
        "pathologies": ["Cancer"],
        "location": ["USA"],
        "size": 13888,
        "modality": "CT",
        "classes": ["Benign", "Malignant"],
    },
    "LiTs": {
        "organ": "Liver",
        "pathologies": ["Cancer"],
        "location": ["Germany", "Netherlands", "Canada", "Israel"],
        "size": 8384,
        "modality": "CT",
        "classes": ["No_Tumor", "Tumor"],
    },
    "RSPECT": {
        "organ": "Lungs",
        "pathologies": ["Pulmonary Embolism"],
        "location": ["Australia", "Turkey", "USA", "Canada", "Brazil"],
        "size": 15000,
        "modality": "CT",
        "classes": ["No_PE", "PE"],
    },
    "IHD_Brain": {
        "organ": "Brain",
        "pathologies": ["Intracranial Hemorrhage"],
        "location": ["USA", "Brazil"],
        "size": 15000,
        "modality": "CT",
        "classes": ["No_IHD", "IHD"],
    },
    "ImageCHD": {
        "organ": "Heart",
        "pathologies": ["Congenital Heart Disease"],
        "location": ["China"],
        "size": 6336,
        "modality": "CT",
        "classes": ["No_CHD", "CHD"],
    },
    "CTPancreas": {
        "organ": "Pancreas",
        "pathologies": ["Pancreatic Cancer"],
        "location": ["USA"],
        "size": 5120,
        "modality": "CT",
        "classes": ["No_Tumor", "Tumor"],
    },
    "Brain_MRI": {
        "organ": "Brain",
        "pathologies": ["Brain Tumour"],
        "location": ["Global"],
        "size": 3160,
        "modality": "MRI",
        "classes": [
            "glioma_tumor",
            "meningioma_tumor",
            "no_tumor",
            "pituitary_tumor",
        ],
    },
    "ProstateMRI": {
        "organ": "Prostate",
        "pathologies": ["Prostate Cancer"],
        "location": ["USA"],
        "size": 2561,
        "modality": "MRI",
        "classes": ["No_Tumor", "Tumor"],
    },
    "RSNAXRay": {
        "organ": "Lungs",
        "pathologies": ["Tuberculosis"],
        "location": ["USA", "China"],
        "size": 801,
        "modality": "Xray",
        "classes": ["No_TB", "TB"],
    },
    "Covid19XRay": {
        "organ": "Lungs",
        "pathologies": ["COVID-19"],
        "location": ["Spain", "USA"], # Check again
        "size": 6057,
        "modality": "Xray",
        "classes": [
            "Negative for Pneumonia",
            "Typical Appearance",
            "Indeterminate Appearance",
            "Atypical Appearance",
        ],
    },
}


class CT_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        name: str,
        num_channels=3,
        mode="Classify",
        train=True,
        preprocess=None,
        transforms=None,
        split_path="./data/CT/splits",
    ) -> None:
        assert mode in MODES
        assert name in DATASETS
        assert os.path.isdir(path)
        super().__init__()

        if train:
            split = "train"
        else:
            split = "val"

        self.mode = mode
        self.path = path
        self.name = name
        self.train = train
        self.num_channels = num_channels
        self.transforms = transforms
        self.preprocess = preprocess

        dt_split_file = os.path.join(split_path, name + "_" + split + ".json")
        dt_split_file = open(dt_split_file, mode="r")
        self.dataset = json.load(dt_split_file)
        assert len(self.dataset["classes"]) == len(self.dataset["inputs"])
        self.classes = set(self.dataset["classes"])
        
    def __len__(self):
        return len(self.dataset["inputs"])

    def get_num_classes(self):
        return len(set(self.dataset["classes"]))


    def __getitem__(self, index):
        img_path = os.path.join(
            self.path, self.name, SLICES_DIR, self.dataset["inputs"][index]
        )
        img = torch.load(img_path)
        img = img.type(torch.FloatTensor)
        if len(img.shape) == 2:
            img = torch.unsqueeze(img, dim=0)
        if img.shape[0] < self.num_channels:
            img = img.repeat(3, 1, 1)
        target = self.dataset["classes"][index]

        mask = None

        if self.preprocess:
            img = self.preprocess(img)

        if self.transforms:
            img = self.transforms(img)
            if mask is not None:
                mask = self.transforms(mask)

        if mask is not None:
            return img, (target, mask)

        return img, target