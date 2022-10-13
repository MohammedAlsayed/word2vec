import json
import gensim
import tqdm
import torch
from smart_open import open

def read_analogies(analogies_fn):
    with open(analogies_fn, "r") as f:
        pairs = json.load(f)
    return pairs


def save_word2vec_format(fname, model, i2v):
    print("Saving word vectors to file...")  # DEBUG
    with open(fname, "wb") as fout:
        fout.write(
            gensim.utils.to_utf8("%d %d\n" % (model.vocab_size, model.embedding_dim))
        )
        # store in sorted order: most frequent words at the top
        for index in tqdm.tqdm(range(len(i2v))):
            word = i2v[index]
            row = model.embed.weight.data[index]
            fout.write(
                gensim.utils.to_utf8(
                    "%s %s\n" % (word, " ".join("%f" % val for val in row))
                )
            )

def get_device(force_cpu, status=True):
    if not force_cpu and torch.backends.mps.is_available():
        device = torch.device("mps")
        if status:
            print("Using mps")
    else:
        device = torch.device("cpu")
        if status:
            print("Using CPU")
    return device