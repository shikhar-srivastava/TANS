from meta.embedding import DatasetEmbeddings
import torch

dataset_embed = DatasetEmbeddings()
data_dict = dataset_embed.parse_and_embed(n_samples = 200)
data_embedding_path = '/nfs/projects/mbzuai/shikhar/datasets/ofa/our_data_path/meta_train.pt'
torch.save(data_dict,data_embedding_path)

for key, value in data_dict.items():
    print(key, len(value['x_query_train']))

print(f'========================== Data Embedding generated! @ {data_embedding_path} DONE ==========================')

