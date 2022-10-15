import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embed(inputs)
        embeds = torch.sum(embeds, dim=1)
        # embeds = torch.mean(embeds, dim=0).reshape((1,-1))
        hidden = F.relu(self.linear1(embeds)) 
        out = self.linear2(hidden)           
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def predict(self,input):
        res = self.forward(input)
        return torch.argmax(res)