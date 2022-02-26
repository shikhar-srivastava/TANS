from functools import reduce
from typing import Union
import json
import os
import pickle

from glob import iglob

import torch
from torch import nn
from meta.ct.loader import DATASETS
CT_DATASETS = DATASETS
XRAY_DATASETS = ['nih', 'chex', 'pc', 'mimic']
ct_model_dict = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'densenet121', 'mobilenetv2', 'mobilenetv3large', 'mobilenetv3small', 'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7']
xray_model_dict = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'densenet121', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']
name_map = {'mobilenetv2': 'mobilenet_v2', 'efficientnetb4': 'efficientnet_b4'}

def get_module_by_name(module: Union[torch.Tensor, nn.Module],
                       access_string: str):
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)

def get_folders(path):
    folders = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            folders.append(dir)
    return folders

def find_best_epoch(data): # XRAY Log format
    best_metric = data[-1]['best_metric']
    best_epoch = -1
    for row in data:
        if row['validauc'] == best_metric:
            best_epoch = row['epoch']
    return best_epoch

class Read_CT_Models():

    def __init__(self, path = None):
        self.path = path if path is not None else '/nfs/projects/mbzuai/BioMedIA/MICCIA_22/logs/classification/CT/post_sanity/'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def parse_model(self, folder, topn):
        # Check if any of the dataset_dict is in the folder name
        for dataset in CT_DATASETS:
            if dataset in folder:
                dataset_name = dataset
                break
        else:
            dataset_name = 'unknown'

        for model in ct_model_dict:
            if model in folder:
                model_name = model
                break
        else:
            model_name = 'unknown'

        if 'without_aug' in folder:
            augmentation = False
        else:
            augmentation = True
        
        if 'balanced' in folder:
            balanced = True
        else:
            balanced = False
        
        if 'pretrained' in folder:
            pretrained = True
        else:
            pretrained = False
        
        if 'batch_128' in folder:
            batch_128 = True
        else:
            batch_128 = False

        files = os.listdir(self.path + topn + '/' + folder)
        meta_data = {} 
        meta_data['best_epoch'] = None
        for file in files:
            # check if file is final.json. If so then load it and check if it has the key 'best_epoch', if so then read 'epoch' from 'best_epoch'
            if file == 'final.json':
                with open(self.path + topn + '/' + folder + '/' + file) as json_file:
                    data = json.load(json_file)
                    if 'best_epoch' in data:
                        meta_data['best_epoch'] = data['best_epoch']['epoch']
                        meta_data['f1'] = data['best_epoch']['f1']
                        meta_data['loss'] = data['best_epoch']['epoch_loss']
                    else:
                        raise Exception(f'final.json in {folder} does not have best_epoch!')
        if  meta_data['best_epoch'] is None:
            raise Exception(f'final.json not present in {folder}!')
        best_model = f'model_epoch_{meta_data["best_epoch"]}.pt'
        meta_data['model_path'] = self.path + topn + '/' + folder + '/' + best_model
        meta_data['dataset'] = dataset_name
        meta_data['model'] = model_name
        meta_data['with_aug'] = augmentation
        meta_data['balanced'] = balanced
        meta_data['pretrained'] = pretrained
        meta_data['batch_128'] = batch_128
        
        #print(meta_data)
        return meta_data
    
    def parse(self, topn):
        folders = get_folders(self.path + topn)
        meta_dict = list()
        for folder in folders:
            data = self.parse_model(folder, topn)
            data['topn'] = topn
            meta_dict.append(data)
        return meta_dict

    def parse_topn(self, standardize = True):
        topns = ['top-2', 'top-4', 'top-7', 'raw']
        meta_dict = list()
        for topn in topns:
            if meta_dict == []:
                meta_dict = self.parse(topn)
            else:
                meta_dict = meta_dict + self.parse(topn)
        if standardize:
            meta_dict = self.standardize(meta_dict)

        return meta_dict
        
    def get_folders(self):
        return self.folders
    
    def get_files(self, folder):
        return os.listdir(self.path + self.topn + '/' + folder)
    
    def test(self):
        return self.parse(testing = True)

    def standardize(self, meta_dict):
        # Standardize model names
        new_dict = list()
        for row in meta_dict:
            if row['model'] in list(name_map.keys()):
                row['model'] = name_map[row['model']]
            new_dict.append(row)
        return new_dict

class Read_XRAY_Models():

    def __init__(self, path = None):
        self.path = path if path is not None else '/nfs/users/ext_shikhar.srivastava/workspace/continualxrayvision/logs/xray/first_run/'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def parse_model(self, folder_path):
        # Check if any of the dataset_dict is in the folder name

        for dataset in XRAY_DATASETS:
            if ('/' + dataset + '/') in folder_path:
                dataset_name = dataset
                break
        else:
            dataset_name = 'unknown'
        for model in xray_model_dict:
            if ('/' + model + '/') in folder_path:
                model_name = model
                break
        else:
            model_name = 'unknown'

        if 'aug_False' in folder_path:
            augmentation = False
        elif 'aug_True' in folder_path:
            augmentation = True
        
        if 'pretrained_False' in folder_path:
            pretrained = False
        elif 'pretrained_True' in folder_path:
            pretrained = True

        files = os.listdir(folder_path)
        meta_data = {}
        meta_data['best_epoch'] = None
        for file in files:
            if 'metrics.pkl' in file:
                with open(folder_path + '/' + file, 'rb') as f:
                    pkl_data = pickle.load(f)
                    best_epoch = find_best_epoch(pkl_data)
                    meta_data['best_epoch'] = best_epoch
                    meta_data['f1'] = pkl_data[-1]['best_metric']
                    meta_data['loss'] = pkl_data[best_epoch-1]['trainloss']
            if 'best.pt' in file:
                meta_data['model_path'] = folder_path + '/' + file

        if  meta_data['best_epoch'] is None:
            print(f'metrics.pkl not present in {folder_path}! Skipping!')
            return None

        meta_data['dataset'] = dataset_name
        meta_data['model'] = model_name
        meta_data['with_aug'] = augmentation
        meta_data['imagenet_pretrained'] = pretrained
        #print(meta_data)
        return meta_data
    
    def parse_subd(self):
        # List all subdirectories two levels down in path 

        level3_subdirectories = iglob(self.path + '/*/*/*')
        meta_dict = list()
        for folder_path in level3_subdirectories:
            model_parse = self.parse_model(folder_path)
            if model_parse is not None:
                meta_dict.append(model_parse)
        return meta_dict
if __name__ == '__main__':
    from torchvision.models import resnet34
    
    model = resnet34()
    get_module_by_name(model, 'layer1.0.relu')


