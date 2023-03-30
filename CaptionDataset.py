import os
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer#T5Tokenizer
from sklearn.model_selection import train_test_split


class CaptionDataset(Dataset):
    def __init__(self, tokenizer, caption_df:pd.DataFrame, task_prefix:str, prepend_prefix:bool = True) -> None:
        self.captions_df = caption_df
        self.task_prefix = task_prefix
        self.prepend_prefix = prepend_prefix
        self.tokenizer = tokenizer

        super().__init__()
    
    def __len__(self):
        return len(self.captions_df)
    

    def get_indices(self, index):
        sub_df = self.captions_df.iloc[index,:]
        if self.prepend_prefix:
            inputs = [self.task_prefix + x for x in sub_df['input']]
        else:
            inputs = list(sub_df['input'])

        outputs = list(sub_df['label'])

        return inputs, outputs

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        inputs, outputs = self.get_indices(index)

        encoding = self.tokenizer(inputs, padding="max_length", max_length=512, truncation=True, return_tensors='pt')
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        target_encoding = self.tokenizer(outputs, padding="max_length", max_length=128, truncation=True, return_tensors='pt')
        labels = target_encoding.input_ids

        # replace padding token id's of the labels by -100 so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return input_ids, attention_mask, labels

class IdxDataset(Dataset):
    def __init__(self, len:int) -> None:
        self.len = len
        super().__init__()
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.tensor(index)

class LoaderWrapper():
    def __init__(self, loader:DataLoader, ds:Dataset) -> None:
        self.loader = loader
        self.ds = ds
    
    def __call__(self):
        for indices in self.loader:
            yield self.ds[indices]


def train_val_test_split(df:pd.DataFrame, train_split:float, val_split:float, test_split:float):
    train_df, other_df = train_test_split(df, train_size=train_split)

    val_split_norm = val_split/(val_split + test_split)
    val_df, test_df = train_test_split(other_df, train_size=val_split_norm)

    return train_df, val_df, test_df


def get_individual_loader(tokenizer, task_prefix:str, prepend_prefix:bool, df:pd.DataFrame, batch_size:int = 64) -> LoaderWrapper:
    ds = CaptionDataset(tokenizer, df, task_prefix, prepend_prefix)
    idx_ds = IdxDataset(len(ds))
    dl_idx = DataLoader(dataset=idx_ds, batch_size=64, shuffle=True)
    loader = LoaderWrapper(dl_idx, ds)
    return loader

def get_loaders(tokenizer, task_prefix:str, prepend_prefix:bool, df:pd.DataFrame, train_split:float, train_batch:int, val_split:float, val_batch:int, test_split:float, test_batch:int):
    train_df, val_df, test_df = train_val_test_split(df, train_split, val_split, test_split)

    trainloader = get_individual_loader(tokenizer, task_prefix, prepend_prefix, train_df, train_batch)
    valloader = get_individual_loader(tokenizer, task_prefix, prepend_prefix, val_df, val_batch)
    testloader = get_individual_loader(tokenizer, task_prefix, prepend_prefix, test_df, test_batch)

    return trainloader, valloader, testloader




if __name__ == "__main__":
    print(torch.rand(2, 3))

    idx = [0, 5, 6, 7, 10, 123, 567, 949]

    df = pd.read_csv('./trainVal_samples.csv')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # print(len(df.iloc[idx,:]))
    # sub = df.iloc[idx,:]
    # print(df.iloc[idx,:])
    # print(list(sub['input']))

    #print(['Tester: '+x for x in sub['input']])




    trainloader, valloader, testloader = get_loaders(tokenizer, 'Tester: ', True, df, train_split=0.6, train_batch=64,
                                                    val_split=0.2, val_batch=64, test_split=0.2, test_batch=64)





    for data in trainloader():
        print(len(data[0]))
        print(data)
        break




    # dl = DataLoader(dataset=ds, batch_size=24)
    # print(len(ds[idx]))

    # for input_ids, attention_mask, labels in dl:
    #     print(len(input_ids), len(attention_mask), len(labels))
    #     break
            