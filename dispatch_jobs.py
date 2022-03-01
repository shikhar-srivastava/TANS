
from re import X
from simple_slurm import Slurm
import os 
import pandas as pd
import time
'''
    Install Slurm library:
        pip install simple-slurm

    *NOTE:
        Incase of Slurm SBATCH overflow, use kill command below:
        > squeue -u ext_shikhar.srivastava | awk '{print $1}' | xargs -n 1 scancel
        # squeue -n ewc_sensitivity | awk '{print $1}' | xargs -n 1 scancel
'''



def dispatch_job(command, params, N = 1 , n = 1, mem = '50G', \
    cpus_per_task = 32, gpus = 4, run_name = 'job',\
        output = '/l/users/shikhar.srivastava/workspace/hover_net/logs/slurm/%j.out', partition = 'mbzuai'):

    '''Dispatch Slurm Job
        Inputs:
        @param command: (str) Python script call command | 
        @param params: (dict) Hypeparameters for the command call | 
    
        Example:
            For command: $python script.py --param1 3.14 --param2 1.618, this will be written as:
            
                command = 'python script.py'
                params['param1'] = 3.14
                params['param2'] = 1.618

                dispatch_job(command, params)
    '''

    print('--- Starting: %s' % run_name)

    slurm = Slurm(N = N, n = n, mem = mem, \
        cpus_per_task = cpus_per_task, gpus=gpus, job_name=run_name,\
            output = output, partition = partition)

    print(slurm)
    for key, value in params.items():
        command += '    --' + str(key) + ' ' + str(value)

    job_id = slurm.sbatch(command, shell ='/bin/bash')
    
    trial_id = '{} > {}'.format(run_name, str(job_id))
    print('Job dispatch details: ', trial_id)
    print(f'command: {command}')


def gen_leave4out():
    MRI = ["Brain_MRI","ProstateMRI"]
    XRAY = ["RSNAXRay","Covid19XRay"]
    CT = [ "MosMed",
    "kits",
    "LiTs",
    "RSPECT",
    "IHD_Brain",
    "ImageCHD",
    "CTPancreas"]
    # get all combinations of 2 CT datasets with 1 MRI and 1 XRAY
    combs = []
    for mri in MRI:
        for xray in XRAY:
            for (ct1,ct2) in combinations(CT, 2):
                combs.append([ct1,ct2,mri,xray])
    return combs
    
from itertools import combinations

if __name__ == '__main__':


    #bucket_step_string = 'top5'
    run_no = 'first_run'

    log_path = f'/nfs/users/ext_shikhar.srivastava/workspace/continualxrayvision/logs/xray/{run_no}/'

    command = "/nfs/users/ext_shikhar.srivastava/miniconda3/envs/remind_proj/bin/python /nfs/users/ext_shikhar.srivastava/workspace/continualxrayvision/pretrain_taskonomy.py"
    gpus = 4
    cpus_per_task = 8*gpus
    mem = str(int(gpus*15)) + 'G'
    partition = 'default-short'
    


    for dataset in datasets:
        for model in models:
            DIR = log_path + dataset + '/' + model

            if not os.path.exists(DIR):
                os.makedirs(DIR)

            for aug in augs:
                for pretraining in imagenet_pretraining:

                    log_folder = f'{DIR}/aug_{aug}-pretrained_{pretraining}/'
                    if not os.path.exists(log_folder):
                        os.makedirs(log_folder)
                
                    output = f'{log_folder}/%j.out'
                    
                    params = dict()
                    if pretraining:
                        params['pretrain'] = ''
                    else:
                        params['no-pretrain'] = ''
                    if aug:
                        params['augs'] = ''
                    else:
                        params['no-augs'] = ''
                    params['model'] = model
                    params['dataset'] = dataset
                    params['output_dir'] = f'{log_folder}'
            
                    print(f'--- Starting: {dataset}, {model}, {aug}, {pretraining}')

                    dispatch_job(command, params, gpus = gpus, cpus_per_task = cpus_per_task, mem = mem, output=output, run_name = dataset+'-'+str(model)+'-'+str(aug)+'-'+str(pretraining), partition = partition)
                    