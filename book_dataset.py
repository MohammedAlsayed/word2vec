from torch.utils.data import Dataset
import pandas as pd
import torch

class CustomImageDataset(Dataset):
    def __init__(self, encoded_sentences, context_size):
        self.encoded_sentences = encoded_sentences
        self.context_size = context_size
        self.counter = 0

    def __len__(self):
        self.num_insts = sum([len(ep)-(2*self.context_size) for ep in self.encoded_sentences])
        return self.num_insts

    def __getitem__(self, idx):
        if idx % self.num_insts == 0:
            self.counter += 1

        sentence = self.encoded_sentences[idx]
        input = torch.from_numpy(self.data.iloc[idx:idx+1 , :-1])
        target = torch.from_numpy(self.data.iloc[idx:idx+1 , -1])
        print(input)
        print(target)
        return input, target