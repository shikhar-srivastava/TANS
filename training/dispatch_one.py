
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

    
from itertools import combinations

def alloc_parition(job_count):
    setup = {'default-short':4, 'multigpu':2, 'default-long':1,}
    n_jobs_avail =  0
    for key, value in setup.items():        
        n_jobs_avail += value
    _index = job_count%n_jobs_avail
    for key, value in setup.items():
        if _index < value:
            return key
        else:
            _index -= value

if __name__ == '__main__':

    run_no = 'first_transfer'
    log_path = f'/nfs/users/ext_shikhar.srivastava/workspace/TANS/training/logs/{run_no}/'
    zero_path = '/nfs/users/ext_shikhar.srivastava/workspace/TANS/training/logs/second_run/'

    command = "/nfs/users/ext_shikhar.srivastava/miniconda3/envs/ofa/bin/python /nfs/users/ext_shikhar.srivastava/workspace/TANS/training/main.py"
    
    gpus = 1
    cpus_per_task = 8*gpus
    mem = str(int(gpus*30)) + 'G'
    job_count = 0
    
    # Get model paths
    subdirs = [os.path.join(zero_path, o) for o in os.listdir(zero_path) if os.path.isdir(os.path.join(zero_path, o))]
    files = []
    for subdir in subdirs:
        files += [os.path.join(subdir,f) for f in os.listdir(subdir) if ((f.endswith('.pt')) & ('metrics' not in f))]

    datasets = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'ImageNet', 'SVHN', 'KMNIST', 'USPS', 'QMNIST']
    
    for model in files:
        for dataset in datasets:
 
            DIR = log_path + model.split('/')[-1].split('.')[0] + '/' + dataset + '/'

            if not os.path.exists(DIR):
                os.makedirs(DIR)
            
            params = dict()
            if 'MNIST' in dataset:
                params['epochs'] = 20
            else:
                params['epochs'] = 50
            params['model']  = model
            params['dataset'] = dataset
            params['log-dir'] = DIR
            output = f'{DIR}/%j.out'
            partition = alloc_parition(job_count)
            print(f'--- Starting: {dataset}, {model}')
            dispatch_job(command, params, gpus = gpus, cpus_per_task = cpus_per_task, mem = mem, output=output, run_name = dataset+'-'+str(model), partition = partition)
            job_count += 1