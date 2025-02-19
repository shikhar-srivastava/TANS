####################################################################################################
# TANS: Task-Adaptive Neural Network Search with Meta-Contrastive Learning
# Wonyong Jeong, Hayeon Lee, Geon Park, Eunyoung Hyung, Jinheon Baek, Sung Ju Hwang
# github: https://github.com/wyjeong/TANS, email: wyjeong@kaist.ac.kr
####################################################################################################

import os
import sys
import glob
import time
import atexit
import torch 
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
#from ofa.model_zoo import ofa_net
from retrieval.loss import HardNegativeContrastiveLoss
from retrieval.measure import compute_recall
import torchvision 

from misc.utils import *
from data.loader import *
from retrieval.nets import *

from sklearn.metrics import f1_score
import random

def remove_module(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k]=v
    return new_state_dict

def set_seed(seed):
    # Set the random seed for reproducible experiments
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class Retrieval:

    def __init__(self, args):
        self.args = args
        self.args.finetuning_lr = 1e-2
        print('Fine-Tuning LR: {}'.format(self.args.finetuning_lr))
        print('args: ', self.args)
        print('===== ============ TRANSFER LEARNING ===========================================================')
        self.parameters = []
        self.device = get_device(args)
        self.cross_trainer = True
        self.mask_self = True
        self.random_pick = False
        self.print_retrievals_only = False
        self.ignore_brainmri = True
        atexit.register(self.atexit)
        set_seed(args.seed)

    def atexit(self):
        print('Process destroyed.')

    def train(self):
        print(f'Begin train process')
        start_time = time.time()
        self.init_loaders()        
        self.init_models()
        self.train_cross_modal_space()
        self.save_cross_modal_space()
        print(f'Process done ({time.time()-start_time:.2f})')
        sys.exit()

    def init_loaders(self):
        print('==> loading data loaders ... ')
        self.tr_dataset, self.tr_loader = get_loader(self.args, mode='train')
        self.te_dataset, self.te_loader = get_loader(self.args, mode='test')
        if self.cross_trainer:
            self.transfer_tr_dataset, self.transfer_tr_loader = get_transfer_loader(self.args, mode='train')
            self.transfer_te_dataset, self.transfer_te_loader = get_transfer_loader(self.args, mode='test')

    def init_models(self):
        print('==> loading encoders ... ')
        self.enc_m = ModelEncoder(self.args).to(self.device)
        self.enc_q = QueryEncoder(self.args).to(self.device)
        self.predictor = PerformancePredictor(self.args).to(self.device)
        self.parameters = [*self.enc_q.parameters(),*self.enc_m.parameters(),*self.predictor.parameters()]
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.args.lr)
        self.criterion = HardNegativeContrastiveLoss(nmax=self.args.n_groups, contrast=True)
        self.criterion_mse = torch.nn.MSELoss()

    
    def train_cross_modal_space(self):
        print('==> train the cross modal space from model-zoo ... ')

        self.scores = {
            'tr_lss': [], 'te_lss': [],
            'r_1': [], 'r_5': [], 'r_10': [], 'r_50': [],
            'mean': [], 'median': [], 'mse': []}
        
        max_recall = 0
        lowest_mse = 1000.0
        lowest_transfer_mse = 1000.0
        start_time = time.time()
        for curr_epoch in range(self.args.n_epochs):
            ep_time = time.time()
            self.curr_epoch = curr_epoch
            
            ##################################################
            self.optimizer.zero_grad()
            dataset, acc, topol, f_emb = next(iter(self.tr_loader)) # 1 iteration == 1 epoch 
            #print(f'dataset: {dataset}, f_emb len: {f_emb.shape}')
            query = self.tr_dataset.get_query(dataset) # Array of queries
            q, m, lss, lss_mse = self.forward(acc, topol, f_emb, query)
            lss = lss + lss_mse
            if self.cross_trainer:
                dataset, acc, f_emb = next(iter(self.transfer_tr_loader))
                #print(f'Transfer> dataset: {dataset}, f_emb len: {f_emb.shape}')
                query = self.transfer_tr_dataset.get_query(dataset)
                transfer_lss_mse = self.transfer_forward(acc, f_emb, query)
                lss = lss + transfer_lss_mse
            lss.backward()
            self.optimizer.step()
            ##################################################
            
            tr_lss = lss.item()
            te_lss, R, medr, meanr, mse = self.evaluate()
            if self.cross_trainer:
                transfer_tr_lss = transfer_lss_mse.item()
                transfer_mse = self.transfer_evaluate()
                #te_lss = te_lss + transfer_te_lss
                print(  f'ep:{self.curr_epoch}, ' +
                    f'transfer_mse:{transfer_mse:.5f} ' +
                    f'transfer_tr_lss: {transfer_tr_lss:.5f}, ' +
                    f'te_lss:{te_lss:.5f}, ' +
                    f'mse:{mse:.5f} ' +
                    f'tr_lss [total train]: {tr_lss:.5f}, ' +
                    f'te_lss:{te_lss:.5f}, ' +
                    f'R@1 {R[1]:.3f} ({max_recall:.2f}), R@5 {R[5]:.2f}, R@10 {R[10]:.2f}, R@50 {R[50]:.2f} ' +
                    f'mean {meanr:.3f}, median {medr:.3f} ({time.time()-ep_time:.3f})')

            else:
                print(  f'ep:{self.curr_epoch}, ' +
                    f'mse:{mse:.5f} ' +
                    f'tr_lss: {tr_lss:.5f}, ' +
                    f'te_lss:{te_lss:.5f}, ' +
                    f'R@1 {R[1]:.3f} ({max_recall:.2f}), R@5 {R[5]:.2f}, R@10 {R[10]:.2f}, R@50 {R[50]:.2f} ' +
                    f'mean {meanr:.3f}, median {medr:.3f} ({time.time()-ep_time:.3f})')

            self.scores['tr_lss'].append(tr_lss)
            self.scores['te_lss'].append(te_lss)
            self.scores['r_1'].append(R[1])
            self.scores['r_5'].append(R[5])
            self.scores['r_10'].append(R[10])
            self.scores['r_50'].append(R[50])
            self.scores['median'].append(medr)
            self.scores['mean'].append(meanr)
            self.scores['mse'].append(mse)
            self.save_scroes()

            if R[1] > max_recall:
                max_recall = R[1]
                self.save_model(True, curr_epoch, R, medr, meanr, mse)

            elif R[1] == max_recall:
                if mse < lowest_mse:
                    lowest_mse = mse
                    self.save_model(True, curr_epoch, R, medr, meanr, mse)
                elif self.cross_trainer and transfer_mse < lowest_transfer_mse:
                    lowest_transfer_mse = transfer_mse
                    self.save_model(True, curr_epoch, R, medr, meanr, mse)
            
        self.save_model(False, curr_epoch, R, medr, meanr, mse)
        self.save_scroes()
        print(f'==> training the cross modal space done. ({time.time()-start_time:.2f}s)')

    def transfer_forward(self, acc, f_emb, query):
         acc = acc.unsqueeze(1).type(torch.FloatTensor).to(self.device)
         query = [d.to(self.device) for d in query]
         q_emb = self.enc_q(query)
         m_emb = self.enc_m(f_emb.to(self.device))
         a_hat = self.predictor(q_emb, m_emb)
         lss_mse = self.criterion_mse(a_hat, acc)
         return lss_mse

    def forward(self, acc, topol, f_emb, query):
        acc = acc.unsqueeze(1).type(torch.FloatTensor).to(self.device)
        query = [d.to(self.device) for d in query]
        q_emb = self.enc_q(query)
        m_emb = self.enc_m(f_emb.to(self.device))
        a_hat = self.predictor(q_emb, m_emb)
        lss = self.criterion(q_emb, m_emb)
        lss_mse = self.criterion_mse(a_hat, acc)
        return q_emb, m_emb, lss, lss_mse

    def evaluate(self):
        dataset, acc, topol, f_emb = next(iter(self.te_loader))
        with torch.no_grad():
            query = self.te_dataset.get_query(dataset)
            q, m, lss, lss_mse = self.forward(acc, topol, f_emb, query)
        recalls, medr, meanr = compute_recall(q.cpu(), m.cpu())
        return lss.item(), recalls, medr, meanr, lss_mse.item()
    
    def transfer_evaluate(self):
        dataset, acc, f_emb = next(iter(self.transfer_te_loader))
        with torch.no_grad():
            query = self.transfer_te_dataset.get_query(dataset)
            lss_mse = self.transfer_forward(acc, f_emb, query)
        return lss_mse.item()

    def save_model(self, is_max=False, epoch=None, recall=None, medr=None, meanr=None, mse=None):
        print('==> saving models ... ')
        if is_max:
            fname = 'saved_model_max_recall.pt'
        else:
            fname = 'saved_model.pt'
        torch_save(self.args.check_pt_path, fname, {
            'enc_q': self.enc_q.cpu().state_dict(),
            'enc_m': self.enc_m.cpu().state_dict(),
            'predictor': self.predictor.cpu().state_dict(),
            'epoch': epoch,
            'recall': recall,
            'medr': medr,
            'meanr': meanr,
            'mse': mse
        })
        self.predictor.to(self.device)
        self.enc_q.to(self.device)
        self.enc_m.to(self.device)

    def save_scroes(self):
        f_write(self.args.log_path, f'cross_modal_space_results.txt', {
            'options': vars(self.args),
            'results': self.scores
        })

    def save_cross_modal_space(self):
        print('==> save the cross modal space from model-zoo ... ')
        self.tr_dataset, self.tr_loader = get_loader(self.args, mode='train')
        self.load_model_zoo()
        self.load_model_encoder()
        self.store_model_embeddings()

    def load_model_zoo(self):
        start_time = time.time()
        self.model_zoo = torch_load(self.args.model_zoo)
        print(f"==> {len(self.model_zoo['dataset'])} pairs have been loaded {time.time()-start_time:.2f}s")

    def load_model_encoder(self):
        print('==> loading model encoder ... ')
        loaded = torch_load(os.path.join(self.args.check_pt_path, 'saved_model_max_recall.pt'))
        self.enc_m = ModelEncoder(self.args).to(self.device)
        self.enc_m.load_state_dict(loaded['enc_m'])
        self.enc_m.eval()

    def store_model_embeddings(self):
        print('==> storing model embeddings ... ')
        start_time = time.time()
        embeddings = {'dataset': [],'m_emb': [], 'f1': [], 'best_epoch': [], 'loss':[], 'model_path':[], 'model':[], 'with_aug':[], 'balanced': [], \
            'pretrained': [], 'batch_128': [] , 'topn': []}
        
        for i, dataset in enumerate(self.model_zoo['dataset']): 
            emb_time = time.time()
            acc = self.model_zoo['f1'][i]
            f_emb = self.model_zoo['f_emb'][i]
            with torch.no_grad():
                m_emb = self.enc_m(f_emb.unsqueeze(0).to(self.device))
            embeddings['dataset'].append(dataset) 
            embeddings['m_emb'].append(m_emb) 
            embeddings['f1'].append(acc)
            #embeddings['f1'].append(self.model_zoo['f1'][i])
            embeddings['loss'].append(self.model_zoo['loss'][i])
            embeddings['model_path'].append(self.model_zoo['model_path'][i])
            embeddings['model'].append(self.model_zoo['model'][i])
            embeddings['with_aug'].append(self.model_zoo['with_aug'][i])
            embeddings['balanced'].append(self.model_zoo['balanced'][i])
            embeddings['pretrained'].append(self.model_zoo['pretrained'][i])
            embeddings['batch_128'].append(self.model_zoo['batch_128'][i])
            embeddings['topn'].append(self.model_zoo['topn'][i])
            embeddings['best_epoch'].append(self.model_zoo['best_epoch'][i])
            
        torch_save(self.args.retrieval_path, 'retrieval.pt', embeddings)
        print(f'==> storing embeddings done. ({time.time()-start_time}s)')

    def test(self):
        print(f'Begin test process')
        start_time = time.time()
        self.init_loaders_for_meta_test()
        print('1')
        self.load_encoders_for_meta_test()
        print('2')
        self.load_cross_modal_space()
        print('3')
        self.meta_test()
        print('4')
        print(f'Process done ({time.time()-start_time:.2f})')

    def init_loaders_for_meta_test(self):
        print('==> loading meta-test loaders')
        self.tr_dataset, self.tr_loader = get_meta_test_loader(self.args, mode='train')
        self.te_dataset, self.te_loader = get_meta_test_loader(self.args, mode='test')

    def load_encoders_for_meta_test(self):
        print('==> loading encoders ... ')
        _loaded = torch_load(os.path.join(self.args.load_path, 'checkpoints', 'saved_model_max_recall.pt'))
        self.enc_q = QueryEncoder(self.args).to(self.device).eval()
        self.enc_q.load_state_dict(_loaded['enc_q'])
        self.predictor = PerformancePredictor(self.args).to(self.device).eval()
        self.predictor.load_state_dict(_loaded['predictor'])

    def load_cross_modal_space(self):
        print('==> loading the cross modal space ... ')
        self.cross_modal_info = torch_load(os.path.join(self.args.load_path, 'retrieval', 'retrieval.pt'))
        self.datasets_cross_modal_info = np.array(self.cross_modal_info['dataset'])
        self.m_embs = torch.stack(self.cross_modal_info['m_emb']).to(self.device)
        
    def meta_test(self):
        print('==> meta-testing on unseen datasets ... ')
        for query_id, query_dataset in enumerate(self.tr_dataset.get_dataset_list()):
            print('New query dataset: ', query_dataset)
            self.tr_dataset.set_dataset(query_dataset)
            self.te_dataset.set_dataset(query_dataset)
            self.query_id = query_id
            self.query_dataset = query_dataset
            self.meta_test_results = {
                'query': self.query_dataset,
                'retrieval': {},
                'random': {},
                'scratch': {},
                'imagenet': {}
            }
            
            query = self.tr_dataset.get_query_set(self.query_dataset)
            query = torch.stack([d.to(self.device) for d in query])
            q_emb = self.get_query_embeddings(query).unsqueeze(0)
            retrieved = self.retrieve(q_emb, self.args.n_retrievals)
            random_retrieved = self.get_random_pick(2) # get 2 randomly retrieved models
            self.dataset_list = []
            self.acc_hat_list = []
            print(f' ========================================================================================================================')
            print(f' [query_id:{query_id+1}] query by {query_dataset} ... ')
            for k, retrieved_dataset in enumerate(retrieved['dataset']):

                m_emb = retrieved['m_emb'][k].to(self.device)
                self.acc_hat = self.predictor(q_emb, m_emb).item()
                self.acc_hat_list.append(self.acc_hat)
                self.dataset_list.append(retrieved_dataset)

            top_j = []
            for j in [int(i) for i in np.argsort(self.acc_hat_list)[-3:-1]]:
                top_j.append(j)
            top_k_idx = top_j
            self.meta_test_results['retrieval_order_asc'] = top_k_idx
            self.meta_test_results['predicted_f1'] = self.acc_hat_list

            self.iterate_and_finetune(retrieved, top_k_idx, type = 'retrieval')
            self.iterate_and_finetune(retrieved, top_k_idx, type = 'scratch')
            self.iterate_and_finetune(retrieved, top_k_idx, type = 'imagenet')
            self.iterate_and_finetune(random_retrieved, np.arange(len(random_retrieved['m_emb'])), type='random')

            self.save_meta_test_results(self.query_dataset)
            del q_emb
            del retrieved
            del random_retrieved
            del self.meta_test_results
        
    def iterate_and_finetune(self, retrieved, idxs, type = 'retrieval'):

        for i, k in enumerate(idxs):
            st = time.time()
            self.i = int(i)
            self.k = int(k)
            
            print(f' >>> Retrieval Type: {type} [Top10 Index - {self.k}((Asc) Retrieval Order - {self.i})]: {retrieved["dataset"][self.k]}, \nmodel: {retrieved["model"][self.k]}, \n\
                with_aug: {retrieved["with_aug"][self.k]}, balanced: {retrieved["balanced"][self.k]},\n\
                        pretrained: {retrieved["pretrained"][self.k]}, batch_128: {retrieved["batch_128"][self.k]}, \n\
                        topn: {retrieved["topn"][self.k]}, best_epoch: {retrieved["best_epoch"][self.k]}, \n\
                            path: {retrieved["model_path"][self.k]},')

            if self.print_retrievals_only:
                continue

            PERCENTAGE_SETS = [0.2, 0.4, 0.6, 0.8, 1.0]
            self.meta_test_results[type][self.k] = {}
            for percentage_samples in PERCENTAGE_SETS:
                self.meta_test_results[type][self.k][f'scores_{int(percentage_samples*100)}'] = {
                    'ep_lss': [],'ep_acc': [], 
                    'ep_tr_time': [],'ep_te_time': [],
                }
            self.meta_test_results[type][self.k]['info'] = {}

            if type == 'retrieval':
                #self.meta_test_results[type][self.k]['scores']['f1_hat'] = self.acc_hat_list[self.k]
                self.meta_test_results[type][self.k]['info'] = {
                    'task': retrieved['dataset'][self.k],'model': retrieved['model'][self.k],'with_aug': retrieved['with_aug'][self.k],
                    'balanced': retrieved['balanced'][self.k],'pretrained': retrieved['pretrained'][self.k],
                    'batch_128': retrieved['batch_128'][self.k],'path': retrieved['model_path'][self.k],
                }
            elif type == 'random':
                self.meta_test_results[type][self.k]['info'] = {
                    'task': retrieved['dataset'][self.k],'model': retrieved['model'][self.k],'with_aug': retrieved['with_aug'][self.k],
                    'balanced': retrieved['balanced'][self.k],'pretrained': retrieved['pretrained'][self.k],
                    'batch_128': retrieved['batch_128'][self.k],'path': retrieved['model_path'][self.k],
                }
            elif type == 'scratch':
                self.meta_test_results[type][self.k]['info'] = {
                    'task': retrieved['dataset'][self.k],'model': retrieved['model'][self.k],'with_aug': 'NA',
                    'balanced': 'NA','pretrained': 'Scratch',
                    'batch_128': 'NA','path': 'NA'
                }
            elif type == 'imagenet':
                self.meta_test_results[type][self.k]['info'] = {
                    'task': retrieved['dataset'][self.k],'model': retrieved['model'][self.k],'with_aug': 'NA',
                    'balanced': 'NA','pretrained': 'ImageNet',
                    'batch_128': 'NA','path': 'NA'
                }
            else:
                raise ValueError('Unknown type in iterate_and_finetune')
                # mse = np.sqrt(np.mean((self.acc_hat-retrieved['f1'][self.k])**2))
                # self.score_list.append(acc)
                # self.mse_list.append(mse)
            
            pre_train_f1 = None
            for percentage_samples in PERCENTAGE_SETS:
                self.model = self.get_model(retrieved['model'][self.k], retrieved['model_path'][self.k], \
                self.tr_dataset.get_n_clss(), type = type)
                self.lss_fn_meta_test = torch.nn.CrossEntropyLoss()
                #self.optim = torch.optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=4e-5)
                self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.finetuning_lr, betas=(0.99, 0.99))
                #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, float(self.args.n_eps_finetuning))
                #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=7, mode='min', factor=0.5, threshold = 1e-4)
                
                if pre_train_f1 is None:
                    pre_train_f1 = self.meta_test_evaluate()[1]
                self.meta_test_results[type][self.k][f'scores_{int(percentage_samples*100)}']['pre_train_f1'] = pre_train_f1
            
                self.tr_dataset.set_dataset(self.query_dataset, percentage_samples = percentage_samples)
                lss, acc = self.fine_tune(self.k, type, percentage_samples)
                
                del self.model
                del self.optim
                del self.lss_fn_meta_test
                #del self.scheduler
            print(' ========================================================================================================================')

    def fine_tune(self, k, type = 'retrieval', percentage_samples = 1.0):
        print(f' ==> finetuning {type}, {self.k}th retreival model ... ')
        self.curr_lr = self.args.lr
        for ep in range(self.args.n_eps_finetuning):
            self.curr_ep = ep
            running_loss = 0.0
            ep_tr_time = 0
            total = 0
            for b_id, batch in enumerate(self.tr_loader):    
                x, y = batch
                b_tr_lss, b_tr_time = self.fine_tune_step(x, y)
                running_loss += b_tr_lss * x.size(0)
                total += x.size(0)
                ep_tr_time += b_tr_time

            tr_lss = running_loss/len(self.tr_dataset) 
            te_lss, te_acc, ep_te_time = self.meta_test_evaluate()
            #self.scheduler.step(te_lss)

            self.meta_test_results[type][self.k][f'scores_{int(percentage_samples*100)}']['ep_tr_time'].append(ep_tr_time)
            self.meta_test_results[type][self.k][f'scores_{int(percentage_samples*100)}']['ep_te_time'].append(ep_te_time)
            self.meta_test_results[type][self.k][f'scores_{int(percentage_samples*100)}']['ep_acc'].append(te_acc)
            self.meta_test_results[type][self.k][f'scores_{int(percentage_samples*100)}']['ep_lss'].append(te_lss)
            
            print(
                f' ==>[r_id:{self.k}({self.i}):]'+
                f' ep:{ep+1}, tr_lss:{tr_lss:.3f},'+
                f' te_lss:{te_lss:.3f}, te_acc: {te_acc:.3f},'+
                f' tr_time:{ep_tr_time:.3f}s, te_time:{ep_te_time:.3f}s')
            
            #self.save_meta_test_results(self.query_dataset)
            if (ep+1)%50==0:
                self.save_meta_test_model(type, percentage_samples)
                print(f'model at ep: {ep} has been saved. ')
        
        return te_lss, te_acc

    def meta_test_evaluate(self):
        total = 0
        crrct = 0 
        ep_time = 0  
        running_loss = 0.0

        prob = []
        labels = []
        preds = []
        for b_id, batch in enumerate(self.te_loader):
            x, y = batch
            labels += y.tolist()
            lss, y_hat, dura = self.meta_test_eval_step(x, y)
            ep_time += dura
            running_loss += lss.item() * x.size(0)
            total += y.size(0)
            prob += (torch.nn.functional.softmax(y_hat, dim=1).detach().cpu().tolist())
            b_preds = torch.argmax(y_hat, dim=-1)
            preds+= b_preds.detach().cpu().tolist()
            #_, y_hat = torch.max(y_hat.data, 1)
            
            #crrct += f1_score(y.to('cpu'), y_hat.to('cpu')).sum().item() #(y_hat == y.to(self.device)).sum().item()
        
        ep_acc = f1_score(labels, preds, average='micro')
        ep_lss = running_loss/len(self.te_dataset) # total 
        return ep_lss, ep_acc, ep_time
        
    def fine_tune_step(self, x, y):
        self.optim.zero_grad()
        self.model.train()
        st = time.time()
        y_hat = self.model(x.to(self.device))
        lss = self.lss_fn_meta_test(y_hat, y.to(self.device))
        lss.backward()
        self.optim.step()
        dura = time.time() - st
        #self.scheduler.step()
        del x
        del y
        return lss.item(), dura

    def meta_test_eval_step(self, x, y):
        st = time.time()
        with torch.no_grad():
            self.model.eval()
            y_hat = self.model(x.to(self.device))
        lss = self.lss_fn_meta_test(y_hat, y.to(self.device))
        dura = time.time() - st
        del x
        del y
        return lss, y_hat, dura

    def get_query_embeddings(self, x_emb):
        print(' ==> converting dataset to query embedding ... ')
        q = self.enc_q(x_emb.unsqueeze(0))
        return q.squeeze()

    def get_model(self, model, model_path, nclss, type = 'retrieval'):

        model_state_dict = torch.load(model_path, map_location=self.device) #TODO: Check if this is loading the state dictionary correctly.
        if model_state_dict.__class__.__name__ == 'DataParallel':
            model_state_dict = model_state_dict.module
        if model_state_dict.__class__.__name__ == 'OrderedDict':
            # Remove DataParallel module from state_dict
            if 'module' in list(model_state_dict.keys())[0]:
                model_state_dict = remove_module(model_state_dict)
                #print('corrected state_dict')
                #print(model.keys())
            if 'resnet' in model:
                final_layer_size = model_state_dict['fc.weight'].shape[0]
            elif 'densenet' in model:
                final_layer_size = model_state_dict['classifier.weight'].shape[0]               
            else:
                final_layer_size = model_state_dict['classifier.1.weight'].shape[0]
            # Load the state_dict into model_obj
            if type == 'retrieval':
                print(f' ==> ({type}) loading model weights ... ')
                model_obj = getattr(torchvision.models, model)(pretrained = False, num_classes = final_layer_size)
                model_obj.load_state_dict(model_state_dict)
            elif type == 'random':
                print(f' ==> ({type}) loading model weights ... ')
                model_obj = getattr(torchvision.models, model)(pretrained = False, num_classes = final_layer_size)
                model_obj.load_state_dict(model_state_dict)
            elif type == 'scratch':
                print(f' ==> ({type}) Not loading model weights ... ')
                model_obj = getattr(torchvision.models, model)(pretrained = False, num_classes = final_layer_size)
            elif type == 'imagenet':
                print(f' ==> ({type}) loading model weights from PyTorch ImageNet pretrained... ')
                model_obj = getattr(torchvision.models, model)(pretrained = True)
            else:
                raise ValueError(f' ==> ({type}) model type not supported. ')
    
            if model_obj.__class__.__name__ == 'ResNet':
                model_obj.fc = torch.nn.Linear(model_obj.fc.in_features, nclss)
                model_obj.fc.weight.data.zero_()
                model_obj.fc.bias.data.zero_()
            
            elif ((model_obj.__class__.__name__ == 'MobileNetV2') | (model_obj.__class__.__name__ == 'EfficientNet')):
                model_obj.classifier[1] = torch.nn.Linear(model_obj.classifier[1].in_features, nclss)
                model_obj.classifier[1].weight.data.zero_()
                model_obj.classifier[1].bias.data.zero_()
            else:
                print(model_obj.__class__.__name__ )
                model_obj.classifier = torch.nn.Linear(model_obj.classifier.in_features, nclss)
                model_obj.classifier.weight.data.zero_()
                model_obj.classifier.bias.data.zero_()
            
            model_obj.to(self.device)
            del model_state_dict #TODO: Check if this is not affecting model_obj.
            return model_obj
            

    def retrieve(self, _q, n_retrieval):
        s_t = time.time()
        scores = torch.mm(_q, self.m_embs.squeeze().t()).squeeze()
        if self.mask_self:
            mask_indices = np.where(self.datasets_cross_modal_info == self.query_dataset)
            scores[mask_indices] = -1.0
        if self.ignore_brainmri: 
            mask_indices = np.where(self.datasets_cross_modal_info == 'Brain_MRI')
            scores[mask_indices] = -1.0
        sorted, sorted_idx = torch.sort(scores, 0, descending=True)
        #sorted_idx[mask_indices] = -1.0
        top_10_idx = sorted_idx[:n_retrieval]
        retrieved = {}
        for idx in top_10_idx:
            for k, v in self.cross_modal_info.items():
                if not k in retrieved:
                    retrieved[k] = []
                # convert idx to int
                retrieved[k].append(v[int(idx.item())])
               
        dura = time.time() - s_t
        self.meta_test_results['retrieval']['search_time'] = dura
        print(f'search time {dura:.5f} s')
        return retrieved
    
    def get_random_pick(self, n_retrieval):

        s_t = time.time()
        # Random array value of shape self.m_embs shape
        scores = torch.tensor(np.random.rand(self.datasets_cross_modal_info.shape[0]))
        if self.mask_self:
            mask_indices = np.where(self.datasets_cross_modal_info == self.query_dataset)
            scores[mask_indices] = -1.0
        if self.ignore_brainmri: 
            mask_indices = np.where(self.datasets_cross_modal_info == 'Brain_MRI')
            scores[mask_indices] = -1.0
        
        sorted, sorted_idx = torch.sort(scores, 0, descending=True)
        #sorted_idx[mask_indices] = -1.0
        top_10_idx = sorted_idx[:n_retrieval]
        retrieved = {}
        for idx in top_10_idx:
            for k, v in self.cross_modal_info.items():
                if not k in retrieved:
                    retrieved[k] = []
                # convert idx to int
                retrieved[k].append(v[int(idx.item())])
        return retrieved

    def get_lr(self):
        for param_group in self.optim.param_groups:
            return param_group['lr']

    def save_meta_test_results(self, query_dataset):
        f_write(self.args.log_path, f'{query_dataset}.txt', {
            'options': vars(self.args),
            'results': self.meta_test_results
        })

    def save_meta_test_model(self, type, percentage_samples):
        torch_save(
            self.args.check_pt_path, 
            f'{self.query_dataset}_{type}_p{int(percentage_samples*100)}_{self.k}_{self.curr_ep}.pt', {
                'model': self.model,
                'curr_ep': self.curr_ep,
                'curr_lr': self.get_lr(), 
        })

