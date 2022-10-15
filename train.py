#!/usr/bin/env python
import time
import argparse
from cProfile import label
from contextlib import ContextDecorator
import os
from venv import create
import tqdm
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from eval_utils import downstream_validation
import utils
import data_utils
from model import CBOW
import matplotlib.pyplot as plt
import torch.nn.functional as F

def create_data_files(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """

    # read in training data from books dataset
    sentences = data_utils.process_book_dir(args.data_dir)

    # build one hot maps for input and output
    (
        vocab_to_index,
        index_to_vocab,
        suggested_padding_len,
    ) = data_utils.build_tokenizer_table(sentences, vocab_size=args.vocab_size)

    # create encoded input and output numpy matrices for the entire dataset and then put them into tensors
    encoded_sentences, lens = data_utils.encode_data(
        sentences,
        vocab_to_index,
        suggested_padding_len,
    )

    # ================== TODO: CODE HERE ================== #
    # Task: Given the tokenized and encoded text, you need to
    # create inputs to the LM model you want to train.
    # E.g., could be target word in -> context out or
    # context in -> target word out.
    # You can build up that input/output table across all
    # encoded sentences in the dataset!
    # Then, split the data into train set and validation set
    # (you can use utils functions) and create respective
    # dataloaders.
    # ===================================================== #

    # create dataset
    context_size = args.context_size
    t = []
    x = []
    counter = 0
    with open(f'./datasets/context_{context_size}_data.txt', 'w') as f:
        for s in tqdm.tqdm(encoded_sentences):
            for index in range(context_size, len(s)-context_size):    
                start = index - context_size
                end = index + context_size + 1
                context = [s[i] for i in range(start, end)if 0 <= i < len(s) and i != index]
                f.write(",".join(str(c) for c in context))
                f.write(f',{s[index]}\n')
                counter +=1
        print("#instances = ",counter)
    

def setup_dataloader(args):
    

     # read in training data from books dataset
    sentences = data_utils.process_book_dir(args.data_dir)

    # build one hot maps for input and output
    (
        vocab_to_index,
        index_to_vocab,
        suggested_padding_len,
    ) = data_utils.build_tokenizer_table(sentences, vocab_size=args.vocab_size)

    # create encoded input and output numpy matrices for the entire dataset and then put them into tensors
    encoded_sentences, lens = data_utils.encode_data(
        sentences,
        vocab_to_index,
        suggested_padding_len,
    )
    context_size = args.context_size
    context = []
    target = []
    for s in tqdm.tqdm(encoded_sentences):
        for idx in range(context_size, len(s)-context_size):
            word_context = []
            for c_idx in range(idx-context_size, idx+context_size+1):
                # c_idx doesn't != target word
                if c_idx != idx:
                    word_context.append(s[c_idx])
            if sum(word_context) == 0 and s[idx] == 0:
                continue
            context.append(word_context)
            target.append(s[idx])

    x = np.asarray(context)
    y = np.asarray(target)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.33, random_state=42)
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

    return train_loader, val_loader, index_to_vocab

def setup_model(args):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your CBOW or Skip-Gram model.
    # ===================================================== #
    model = CBOW(args.vocab_size, args.embedding_dim)
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for predictions. 
    # Also initialize your optimizer.
    # ===================================================== #
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True,
):
    model.train()
    epoch_loss = 0.0

    # keep track of the model predictions for computing accuracy
    pred_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in tqdm.tqdm(loader):
        # put model inputs to device
        inputs, labels = inputs.to(device).long(), labels.to(device).float()
        labels = F.one_hot(labels.to(torch.int64), num_classes=args.vocab_size)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        pred_logits = model(inputs)
        
        # calculate prediction loss
        loss = criterion(pred_logits, labels.float())

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_loss += loss.item()

        # compute metrics
        preds = pred_logits.argmax(-1)
        pred_labels.extend(preds.cpu().numpy())
        target = labels.argmax(-1)
        target_labels.extend(target.cpu().numpy())

    acc = accuracy_score(pred_labels, target_labels)
    epoch_loss /= len(loader)

    return epoch_loss, acc


def validate(args, model, loader, optimizer, criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

    return val_loss, val_acc


def main(args):

    device = utils.get_device(args.force_cpu)

    # load analogies for downstream eval
    external_val_analogies = utils.read_analogies(args.analogies_fn)

    if args.downstream_eval:
        word_vec_file = os.path.join(args.outputs_dir, args.word_vector_fn)
        assert os.path.exists(word_vec_file), "need to train the word vecs first!"
        downstream_validation(word_vec_file, external_val_analogies)
        return

    # create_data_files(args)
    # get dataloaders
    train_loader, val_loader, i2v = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args)
    model.to(device)
    print(model)

    # # get optimizer
    criterion, optimizer = setup_optimizer(args, model)

    train_accuracies = []
    train_losses = []

    val_accuracies = []
    val_losses = []
    train_epochs = [i for i in range(1, args.num_epochs+1)]
    val_epochs = []

    st = time.time()
    for epoch in range(args.num_epochs):
        # train model for a single epoch
        print(f"Epoch {epoch}")
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )
        train_accuracies.append(train_acc)
        train_losses.append(train_losses)
        print(f"train loss : {train_loss} | train acc: {train_acc}")

        if epoch % args.val_every == 0:
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device
            )
            val_epochs.append(epoch)
            val_accuracies.append(val_acc)
            val_losses.append(val_loss)
            print(f"val loss : {val_loss} | val acc: {val_acc}")

            # ======================= NOTE ======================== #
            # Saving the word vectors to disk and running the eval
            # can be costly when you do it multiple times. You could
            # change this to run only when your training has concluded.
            # However, incremental saving means if something crashes
            # later or you get bored and kill the process you'll still
            # have a word vector file and some results.
            # ===================================================== #
        # save the plots 
        if epoch == args.num_epochs-1:
            print("plotting training")
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            
            ax[0].plot(train_epochs, train_accuracies, 'b', label='training accuracy')
            ax[0].set_title('Training Accuracy')
            ax[0].set_xlabel('Epochs')
            ax[0].set_ylabel('Accuracy')
            ax[0].legend()

            ax[1].plot(train_epochs, train_accuracies, 'r', label='training loss')
            ax[1].set_title('Training loss')
            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Loss')
            ax[1].legend()
            fig.savefig(f'./plots/train_acc_loss_context_{args.context_size}.png')

            print("plotting validation")
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(val_epochs, val_accuracies, 'b', label='validation accuracy')
            ax[0].set_title('Validation Accuracy')
            ax[0].set_xlabel('Epochs')
            ax[0].set_ylabel('Accuracy')
            ax[0].legend()

            ax[1].plot(val_epochs, val_losses, 'r', label='validation loss')
            ax[1].set_title('Validation loss')
            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Loss')
            ax[1].legend()
            fig.savefig(f'./plots/validation_acc_loss_context_{args.context_size}.png')

            train_end_time = time.time()
            print("training time = ", train_end_time-st , " sec")

        # save word vectors in last epoch
        if epoch == args.num_epochs-1:
            w2v_st = time.time()
            word_vec_file = f'./results/learned_word_vectors_context_{args.context_size}.txt'
            print("saving word vec to ", word_vec_file)
            utils.save_word2vec_format(word_vec_file, model, i2v)
            w2v_et = time.time()
            print("time to save word2vec = ", w2v_et-w2v_st , " sec")

            downstream_st = time.time()
            # evaluate learned embeddings on a downstream task
            downstream_validation(word_vec_file, external_val_analogies)
            downstream_et = time.time()
            print("downstream time = ", downstream_et-downstream_st , " sec")

        # save model in last epoch
        if epoch == args.num_epochs-1:
            ckpt_file = os.path.join(args.output_dir, f'model_context_{args.context_size}.ckpt')
            print("saving model to ", ckpt_file)
            torch.save(model, ckpt_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="where to save training outputs")
    parser.add_argument("--data_dir", type=str, help="where the book dataset is stored")
    parser.add_argument(
        "--downstream_eval",
        action="store_true",
        help="run downstream eval on trained word vecs",
    )
    # ======================= NOTE ======================== #
    # If you adjust the vocab_size down below 3000, there 
    # may be analogies in the downstream evaluation that have
    # words that are not in your vocabulary, resulting in
    # automatic (currently) zero score for an ABCD where one
    # of A, B, C, or D is not in the vocab. A visible warning
    # will be generated by the evaluation loop for these examples.
    # ===================================================== #
    parser.add_argument(
        "--vocab_size", type=int, default=3000, help="size of vocabulary"
    )
    parser.add_argument(
        "--embedding_dim",
        default=128,
        type=int,
        help="size of word embedding dimension",
    )
    parser.add_argument(
        "--context_size",
        default=1,
        type=int,
        help="context size of the target word",
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument(
        "--analogies_fn", type=str, help="filepath to the analogies json file"
    )
    parser.add_argument(
        "--word_vector_fn", type=str, help="filepath to store the learned word vectors",
        default='learned_word_vectors.txt'
    )
    parser.add_argument(
        "--num_epochs", default=7, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--val_every",
        default=3,
        type=int,
        help="number of epochs between every eval loop",
    )
    parser.add_argument(
        "--save_every",
        default=5,
        type=int,
        help="number of epochs between saving model checkpoint",
    )
    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #

    args = parser.parse_args()
    main(args)
