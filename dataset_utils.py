import random
import numpy as np
import torch
from torch.utils.data import Dataset, RandomSampler, BatchSampler


random_seed = 8138
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU


class default_dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {"input" : self.data[idx], "label" : self.label[idx]}


# define variable length data
class variable_dataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return {"input":torch.tensor([idx] * (idx+1)), 
                "label": torch.tensor(idx)}


if __name__ == "__main__":
    input_data = torch.tensor([[i, i+1, i+2] for i in range(0, 10)])
    label_data = torch.tensor([i for i in range(0, 10)]) 

    dataset = default_dataset(input_data, label_data)

    """DEFAULT SETTING"""
    dataloader = torch.utils.data.DataLoader(dataset) # batch O
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4) # batch X
   
    
    """RANDOM SAMPLER""" 
    random_sampler = RandomSampler(dataset) 
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, sampler=random_sampler)
    dataloader2 =torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)


    """BATCH SAMPLER"""
    random_sampler = RandomSampler(dataset)
    batch_sampler = BatchSampler(random_sampler, batch_size = 3, drop_last=False) #include batch_size, shuffle, drop_last
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler) 

    # for data in dataloader:
    #     print(data["input"], data["label"])

    """COLLATE FUNCTION""" # -> use when dataset is variable length 
    # var_dataset = variable_dataset()
    
    # dataloader = torch.utils.data.DataLoader(var_dataset)
    # for data in dataloader:
    #     print(data['input'])


    # # ERROR
    # dataloader = torch.utils.data.DataLoader(var_dataset, batch_size=2)
    # for data in dataloader:
    #     print(data['input'].shape, data['label'])


