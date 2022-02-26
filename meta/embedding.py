import torch
import torch.nn.functional as F
import torchvision

from meta.utils import get_module_by_name, Read_CT_Models, Read_XRAY_Models
from meta.ct.loader import DATASETS, CT_Dataset
import torchvision.transforms as T
import torchvision
from meta.ct.transforms import compose, Clip
import torchxrayvision as xrv
import tqdm 

import pandas as pd

CT_DATASETS = DATASETS
XRAY_DATASETS = ['nih', 'chex', 'pc', 'mimic']
nih_path = '/nfs/projects/mbzuai/shikhar/datasets/nih/images'
chexpert_path = '/nfs/projects/mbzuai/shikhar/datasets/chexpert_small/CheXpert-v1.0-small'
padchest_path = '/nfs/projects/mbzuai/shikhar/datasets/padchest'
mimic_path = '/nfs/projects/mbzuai/shikhar/datasets/mimic/physionet.org/files/mimic-cxr-jpg/2.0.0'


our_m_train_path = '/nfs/projects/mbzuai/shikhar/datasets/ofa/our_m_train.pt'

def sanity_model_parse(all_models):
    print('=====================')
    print('Sanity Model Parse')
    df = pd.DataFrame(all_models)
    # Check for unknowns
    unknowns = df[(df['dataset'] == 'unknown') | (df['model'] == 'unknown')]
    print('Unknowns: ', len(unknowns))
    print(unknowns)
    print('=====================')
    return unknowns

def remove_module(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k]=v
    return new_state_dict
    #model.load_state_dict(new_state_dict)

class ModelEmbeddings():

    def __init__(self, \
        layer = 'pre_final', \
        noise_path = '/nfs/users/ext_shikhar.srivastava/workspace/TANS/noise.pt',\
        reduce = True):
        
        self.noise = torch.load(noise_path)
        self.activations = {}
        self.embed_layer_dict = {'ResNet': 'fc', 'DenseNet':'classifier', 'EfficientNet': 'classifier', 'MobileNetV2':'classifier', 'MobileNetV3':'classifier'} if layer == 'final' \
            else \
                {'ResNet': 'avgpool',  'DenseNet':'features', 'EfficientNet':'avgpool', 'MobileNetV2':'features', 'MobileNetV3':'features'}
        self.reduce = reduce
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def f_embed(self, model):
        #=============================================================================
        # Hook function to get the activations of the model
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        #=============================================================================
        # Get the embeddings
        model.eval()
        with torch.no_grad():
             # Register hooks on the model
            try:
                handler = get_module_by_name(model, self.embed_layer_dict[model.__class__.__name__]).register_forward_hook(get_activation(self.embed_layer_dict[model.__class__.__name__]))
            except Exception as e:
                print(e,'[Activation Fetching Error]')
                return None

            _ = model(self.noise)
            f_emb = self.activations[self.embed_layer_dict[model.__class__.__name__]]
            handler.remove() # Remove the hook

        if (self.reduce) & (f_emb.shape[-2:] != (1,1)):
            f_emb = F.adaptive_avg_pool2d(f_emb, (1, 1))
            
        return f_emb.flatten()
    
    def parse_and_embed(self, standard_size = 2048, no_xray = True):
        # Read and collate Model list
        if no_xray:
            ct_reader = Read_CT_Models()
            ct_models = ct_reader.parse_topn()
            all_models = ct_models.copy()
            print('No Xrays!')
        else:    
            xray_reader = Read_XRAY_Models()
            ct_reader = Read_CT_Models()
            xray_models = xray_reader.parse_subd()
            ct_models = ct_reader.parse_topn()
            all_models = xray_models.copy() + ct_models.copy()
        
        s_check = sanity_model_parse(all_models) # Sanity
        if len(s_check) > 0:
            print('[Model Sanity Check] not passed.', s_check)
            return s_check
        
        
        _all_models = []
        for meta_dict in all_models:
            try:
                model = torch.load(meta_dict['model_path'], map_location=self.device)
                if model.__class__.__name__ == 'DataParallel':
                    model = model.module
                if model.__class__.__name__ == 'OrderedDict':
                    # Remove DataParallel module from state_dict
                    if 'module' in list(model.keys())[0]:
                        model = remove_module(model)
                        #print('corrected state_dict')
                        #print(model.keys())
                    if 'resnet' in meta_dict['model']:
                        final_layer_size = model['fc.weight'].shape[0]
                    elif 'densenet' in meta_dict['model']:
                        final_layer_size = model['classifier.weight'].shape[0]               
                    else:
                        final_layer_size = model['classifier.1.weight'].shape[0]
                    # Load the state_dict into model_obj
                    model_obj = getattr(torchvision.models, meta_dict['model'])(pretrained = False, num_classes = final_layer_size)
                    model_obj.load_state_dict(model)       
                    model = model_obj

            except Exception as e:
                print(e)
                print('[Model Loading Error]')
                print(meta_dict['model_path'])
                #print(model.__class__.__name__)
                print(meta_dict['model'])
                continue
            
            f_emb = self.f_embed(model)

            meta_dict['f_emb']  = torch.cat((f_emb, torch.zeros(standard_size - f_emb.shape[0])), dim = 0)
            del model
            _all_models.append(meta_dict)
        
        return _all_models




class DatasetEmbeddings():

    def __init__(self, category = 'raw'):
        self.category = category
        self.resnet = torchvision.models.resnet18(pretrained=True)
        

    def get_dataset_object(self, dataset, type = 'CT', num_channels = 1, train_split = True):

        assert type in ['CT', 'XRAY', 'MRI'], 'Dataset type not supported'
        if type == 'CT':
            assert dataset in CT_DATASETS, f'{dataset} not in {CT_DATASETS}'
            # config params
            path = '/nfs/projects/mbzuai/BioMedIA/MICCIA_22/Taskonomy_preprocessed/'
            
            train_split_path = f'/nfs/users/ext_shikhar.srivastava/workspace/MedicalTaskonomy//data/CT/splits/balanced/{self.category}'

            transforms = T.Compose([Clip([-1000, 1000]), T.Normalize((0.5,),(0.5,))])
            dataset_object = CT_Dataset(
                path=path,
                name=dataset,
                train=train_split,
                num_channels=num_channels,
                preprocess=transforms,
                split_path=train_split_path,
            )
            return dataset_object
        elif type == 'XRAY':
            assert dataset in XRAY_DATASETS, f'{dataset} not in {XRAY_DATASETS}'
            transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
            if "chex" in dataset:
                dataset_object = xrv.datasets.CheX_Dataset(
                    imgpath=chexpert_path,
                    csvpath=chexpert_path +"/train.csv",
                    transform=transforms, data_aug=None, unique_patients=False)
                return dataset_object
            elif "nih" in dataset:
                dataset_object = xrv.datasets.NIH_Dataset(
                    imgpath=nih_path, 
                    transform=transforms, data_aug=None, unique_patients=False)
                return dataset_object
            elif "pc" in dataset:
                dataset_object = xrv.datasets.PC_Dataset(
                    imgpath=padchest_path,
                    transform=transforms, data_aug=None, unique_patients=False)
                return dataset_object
            elif "mimic" in dataset:
                dataset_object = xrv.datasets.MIMIC_Dataset(
                imgpath=mimic_path + '/files',
                csvpath=mimic_path +'/mimic-cxr-2.0.0-metadata.csv.gz',
                metacsvpath=mimic_path +'/mimic-cxr-2.0.0-chexpert.csv.gz',
                transform=transforms, data_aug=None, unique_patients=False)
                return dataset_object

    
    def x_embedding(self, x):
        activations = {}
         #=============================================================================
        # Hook function to get the activations of the model
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        #=============================================================================
        # Get the embeddings
        self.resnet.eval()
        with torch.no_grad():
             # Register hooks on the model
            try:
                handler = self.resnet.avgpool.register_forward_hook(get_activation('avgpool'))
            except Exception as e:
                print(e,'[Activation Fetching Error]')
                return None

            _ = self.resnet(x)
            f_emb = activations['avgpool']
            handler.remove() # Remove the hook
        return f_emb.squeeze()

    def embed_dataset_single(self, dataset, n_samples):
        dataset_object = self.get_dataset_object(dataset, num_channels=3)
        train_loader = torch.utils.data.DataLoader(dataset_object, batch_size=n_samples, shuffle=True)
        x_train, _ = train_loader.__iter__().__next__()
        embed = self.x_embedding(x_train)
        return embed

    def meta_test_embed_datasets(self, n_samples):
        data_train = {}
        for dataset in tqdm.tqdm(list(set(CT_DATASETS))):
            query_embed = self.embed_dataset_single(dataset, n_samples=n_samples)
            task = dataset
            clss = self.get_dataset_object(dataset, type = 'CT').classes
            nclss = len(clss)
            data_train[task] = {'task': task, 'clss': clss, 'nclss': nclss, 'query': query_embed}
        return data_train

    def embed_dataset(self, dataset, type, n_samples):
        train_dataset_object = self.get_dataset_object(dataset, type, train_split = True)
        test_dataset_object = self.get_dataset_object(dataset, type, train_split=False)
        train_loader = torch.utils.data.DataLoader(
                                        train_dataset_object,
                                        batch_size=n_samples,
                                        shuffle=True
                                    )
        test_loader = torch.utils.data.DataLoader(
                                        test_dataset_object,
                                        batch_size=n_samples,
                                        shuffle=True
                                    )
        
        if type == 'CT':
            x_train,y_train = train_loader.__iter__().__next__()
            x_test, y_test = test_loader.__iter__().__next__()
            del train_loader
            del test_loader
        elif type == 'XRAY':
            _train = train_loader.__iter__().__next__()
            _test = test_loader.__iter__().__next__()
            del train_loader
            x_train = _train['img']
            x_test = _test['img']
            y_train = _train['lab']
            y_test = _test['lab']

        x_train_embed = []
        x_test_embed = []
        for _x in x_train:
            x_train_embed.append(self.x_embedding(_x.repeat(1, 3, 1, 1)))
        for _x in x_test:
            x_test_embed.append(self.x_embedding(_x.repeat(1, 3, 1, 1)))

        return x_train_embed, y_train, x_test_embed, y_test

    def parse_and_embed(self, n_samples, no_xray = True):
        if no_xray:
            datasets = list(set(CT_DATASETS))
        else:
            datasets = list(set(XRAY_DATASETS + CT_DATASETS) - set(['mimic']))
        data_train = {}
        # use tqdm to iterate
        for dataset in tqdm.tqdm(datasets):
            print(dataset)
            if dataset in XRAY_DATASETS:
                x_train_embed, y_train, x_test_embed, y_test = self.embed_dataset(dataset = dataset, type = 'XRAY', n_samples = n_samples)
                task = dataset
                clss = self.get_dataset_object(dataset, type = 'XRAY').pathologies
                nclss = len(clss)
                data_train[dataset] = {'task': task, 'clss': clss, 'nclss':nclss, 'x_query_train':x_train_embed, 'y_query_train':y_train, \
                    'x_query_test':x_test_embed, 'y_query_test':y_test, 'type': 'XRAY', 'task_type':'classification'}
            else:
                x_train_embed, y_train, x_test_embed, y_test = self.embed_dataset(dataset = dataset, type = 'CT', n_samples = n_samples)
                task = dataset
                clss = self.get_dataset_object(dataset, type = 'CT').classes
                nclss = len(clss)
                data_train[dataset] = {'task': task, 'clss': clss, 'nclss':nclss, 'x_query_train':x_train_embed, 'y_query_train':y_train, \
                    'x_query_test':x_test_embed, 'y_query_test':y_test, 'type': 'CT', 'task_type':'classification'}
        return data_train


    '''def parse_and_write_dataset_samples(self):
        datasets = list(set(CT_DATASETS))
        data={} 
        #dict_keys(['task', 'clss', 'nclss', 'x_train', 'y_train', 'x_test', 'y_test', 'query'])
        for dataset in tqdm.tqdm(datasets):
            train_dataset_object = self.get_dataset_object(dataset, type, num_channels=3, train_split = True)
            train_loader = torch.utils.data.DataLoader(
                                        train_dataset_object, 
                                        batch_size=self.n_samples,      
                                        shuffle=True
                                    )
            test_dataset_object = self.get_dataset_object(dataset, type, num_channels=3, train_split = False)
            test_loader = torch.utils.data.DataLoader(
                                        test_dataset_object,    
                                        batch_size=self.n_samples,  
                                        shuffle=True
                                    )'''
            



if __name__ == '__main__':

    # Test out Embeddings for set of models below
    import torchvision.models as models
    nets = {'resnet18': models.resnet18(), 'resnet34': models.resnet34(), 'resnet50': models.resnet50(), 'resnet101': models.resnet101(), 'densenet121': models.densenet121(), \
        
        'mobilenet_v2': models.mobilenet_v2(), 'mobilenet_v3_large': models.mobilenet_v3_large(), 'mobilenet_v3_small': models.mobilenet_v3_small(), \
            
        'efficientnet_b0': models.efficientnet_b0(), 'efficientnet_b1': models.efficientnet_b1(), 'efficientnet_b2': models.efficientnet_b2(), \

        'efficientnet_b3': models.efficientnet_b3(), 'efficientnet_b4': models.efficientnet_b4(), 'efficientnet_b5': models.efficientnet_b5(), 'efficientnet_b6': models.efficientnet_b6(), 'efficientnet_b7': models.efficientnet_b7()}

    embed = ModelEmbeddings(layer = 'pre_final', reduce = True)
    f_emb = {}
    for key, value in nets.items():
        print(key)
        f_emb[key] = embed.f_embed(value)

    for key, value in f_emb.items():
        print(key, value.shape)
    
    print('Success!')