####################################################################################################
# TANS: Task-Adaptive Neural Network Search with Meta-Contrastive Learning
# Wonyong Jeong, Hayeon Lee, Geon Park, Eunyoung Hyung, Jinheon Baek, Sung Ju Hwang
# github: https://github.com/wyjeong/TANS, email: wyjeong@kaist.ac.kr
####################################################################################################

import os
import glob
import torch
import time
import random
import numpy as np
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from misc.utils import *
import torchvision.transforms as T
from meta.ct.transforms import compose, Clip

def get_loader(args, mode='train'):
    dataset = MetaTrainDataset(args, mode=mode)
    loader = DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        shuffle=(mode=='train'),
                        num_workers=4)
    return dataset, loader

def get_meta_test_loader(args, mode='train'):
    dataset = MetaTestDataset(args, mode=mode)
    loader = DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        shuffle=(mode=='train'),
                        num_workers=4)
    return dataset, loader

MODES = ["Classify", "Segment"]
SLICE_EXT = ".pt"
SLICES_DIR = "slices"


DATASETS = [
    "MosMed",
    "kits",
    "LiTs",
    "RSPECT",
    "IHD_Brain",
    "ImageCHD",
    "CTPancreas",
    "Brain_MRI",
    "ProstateMRI",
    "RSNAXRay",
    "Covid19XRay",
]

# OUT = ['MosMed', 'RSNAXRay', 'Brain_MRI']
# IN = ['Covid19XRay',]
class MetaTestDataset(Dataset):
    def __init__(self, args, dataset_list = ['MosMed', 'kits', 'Brain_MRI', 'RSNAXRay'], mode='train'):
        self.args = args
        self.mode = mode
        self.path = '/nfs/projects/mbzuai/BioMedIA/MICCIA_22/Taskonomy_preprocessed/'
        self.dataset_list = dataset_list
        self.dataset_emb = torch_load(os.path.join(self.args.data_path, 'meta_test_all.pt'))
        #self.data = torch_load(os.path.join(self.args.data_path, f'meta_test_{self.dataset_list[0]}.pt'))
        self.curr_dataset = self.dataset_list[0]
        self.split_path = "/nfs/users/ext_shikhar.srivastava/workspace/MedicalTaskonomy/data/CT/splits/balanced/raw"
        self.mode = mode
        if mode == 'train':
            self.split = "train"
        else:
            self.split = "val"
        self.num_channels = 3
        self.transforms =  T.Compose([Clip([-1000, 1000]), T.Normalize((0.5,),(0.5,))])
    
        dt_split_file = os.path.join(self.split_path, self.curr_dataset + "_" + self.split + ".json")
        dt_split_file = open(dt_split_file, mode="r")
        self.dataset_data = json.load(dt_split_file)

        self.dataset_cache = self.cache_dataset()

    
    def cache_dataset(self):
        dataset = {}
        for index in range(len(self.dataset_data["inputs"])):

            img_path = os.path.join(
                self.path, self.curr_dataset, SLICES_DIR, self.dataset_data["inputs"][index]
            )
            img = torch.load(img_path)
            img = img.type(torch.FloatTensor)
            if len(img.shape) == 2:
                img = torch.unsqueeze(img, dim=0)
            if img.shape[0] < self.num_channels:
                img = img.repeat(3, 1, 1)
            target = self.dataset_data["classes"][index]

            dataset[index] = (img, target)
        return dataset

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.split = "train"
        else:
            self.split = "val"
        dt_split_file = os.path.join(self.split_path, self.curr_dataset + "_" + self.split + ".json")
        dt_split_file = open(dt_split_file, mode="r")
        self.dataset_data = json.load(dt_split_file)

    def get_dataset_list(self):
        return self.dataset_list

    def set_dataset(self, dataset):
        del self.dataset_data
        del self.dataset_cache
        self.curr_dataset = dataset
        dt_split_file = os.path.join(self.split_path, self.curr_dataset + "_" + self.split + ".json")
        dt_split_file = open(dt_split_file, mode="r")
       
        self.dataset_data = json.load(dt_split_file)
        print(f"{dataset}: #_train: {len(self.dataset_data['inputs'])}")

        self.dataset_cache = self.cache_dataset()

    def __len__(self):
        return len(self.dataset_data['inputs'])

    def __getitem__(self, index):
        try:
            img, target = self.dataset_cache[index]
        except KeyError as k_error:
            print("Not found in cache: %s" % self.dataset_data["inputs"][index])
            raise KeyError from k_error

        if self.transforms:
            img = self.transforms(img)

        return img, target
        

    def get_query_set(self, task):
        
        return self.dataset_emb[task]['query']

    def get_n_clss(self):
        return len(set(self.dataset_data["classes"]))

class MetaTrainDataset(Dataset):

    def __init__(self, args, mode='train'):
        start_time = time.time()
        self.args = args
        self.mode = mode
        self.model_zoo = torch_load(self.args.model_zoo)
        self.query = torch_load(os.path.join(self.args.data_path, 'meta_train.pt'))
        start_time = time.time()
        self.contents = []
        self.dataset_list = set(self.model_zoo['dataset'])
        print('Length of model_zoo: ', len(self.model_zoo))

        for dataset in self.dataset_list:
            models = []
            cnt = 0
            for idx, _dataset in enumerate(self.model_zoo['dataset']):
                if dataset == _dataset:
                    cnt+= 1
                    if cnt <= self.args.n_nets:
                        '''############################################
                        topol = self.model_zoo['topol'][idx]
                        ks = topol[:20] 
                        e = topol[20:40]
                        d = topol[40:]
                        tmp = torch.zeros(len(ks))
                        for stage, num_layer in enumerate(d):
                            tmp[stage*4:stage*4+num_layer] = 1
                        ks = torch.tensor(ks) * tmp
                        e = torch.tensor(e) * tmp
                        topol = [int(t) for t in [*ks.tolist(), *e.tolist(), *d]]
                        ############################################'''
                        
                        models.append({'f1': self.model_zoo['f1'][idx],
                            'topol': [-1], #self.model_zoo['topol'][idx],
                            'f_emb': self.model_zoo['f_emb'][idx],
                            'n_params': [-1], #self.model_zoo['n_params'][idx],
                        })
            self.contents.append((dataset, models))
        #print('Length of contents: ', len(self.contents))
        #print(self.contents)
        #print(f"{len(self.contents)*self.args.n_nets} pairs loaded ({time.time()-start_time:.3f}s) ")

    def __len__(self):
        return len(self.contents) 

    def set_mode(self, mode):
        self.mode = mode
        
    def __getitem__(self, index):
        dataset = self.contents[index][0]
        n_models = len(self.contents[index][1])
        if n_models == 1:
            idx = 0 
        else:
            idx = random.randint(0,n_models-1)
        model = self.contents[index][1][idx]
        acc = model['f1'] 
        n_params = model['n_params']
        topol = torch.Tensor(model['topol'])
        f_emb = model['f_emb']
        return dataset, acc, topol, f_emb

    def get_query(self, datasets):
        x_batch = []
        for d in datasets:
            x = self.query[d][f'x_query_{self.mode}']
            x_batch.append(torch.stack(x))
        return x_batch





    