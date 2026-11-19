import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

# we load one shared tokenizer for the whole file, so we do not load it again
# every time we build a dataset
tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

# t5 has no real beginning of sentence token, so we use one of its spare extra ids
# as our bos. we use it as the first token of the decoder input in training,
# and as the first token when we start generating at eval time,
# so training and generation always match.
BOS_ID = tokenizer.convert_tokens_to_ids("<extra_id_0>")

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose. 
            * Class behavior should be different on the test set.
        '''
        # we remember the split because the test set behaves differently
        self.split = split
        # tokenize everything once here, not every time we take one example
        self.process_data(data_folder, split, tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # tokenize all the natural language questions
        self.encoder_ids = [
            tokenizer(line)["input_ids"]
            for line in load_lines(os.path.join(data_folder, split + ".nl"))
        ]

        # now the sql targets, but the test set has no sql file, the labels are
        # hidden for the leaderboard. also, the tokenizer adds the end of sentence
        # token </s> to the sql by itself and we keep it, because it is the stop
        # signal the model has to learn to predict.
        self.decoder_ids = []
        if split != "test":
            self.decoder_ids = [
                tokenizer(line)["input_ids"]
                for line in load_lines(os.path.join(data_folder, split + ".sql"))
            ]

    def __len__(self):
        # one example per line of the .nl file
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        # on test we return only the question tokens, there are no sql targets
        if self.split == "test":
            return torch.tensor(self.encoder_ids[idx], dtype=torch.long)
        # on train and dev we return the question tokens and the sql target tokens
        return (
            torch.tensor(self.encoder_ids[idx], dtype=torch.long),
            torch.tensor(self.decoder_ids[idx], dtype=torch.long),
        )


def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # the batch is a list of (question, sql) pairs, so we split it into two lists
    encoder_list = [item[0] for item in batch]
    target_list = [item[1] for item in batch]

    # pad the questions to the longest question in this batch,
    # and build the mask, 1 where the token is real, 0 where it is pad
    encoder_ids = pad_sequence(encoder_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    # here is the teacher forcing trick. the decoder input is the target shifted
    # right: BOS + all target tokens except the last one. this way at every position
    # the decoder only sees the tokens before position t and predicts the token
    # at position t. it never gets to peek at the answer.
    decoder_input_list = [
        torch.cat([torch.tensor([BOS_ID]), target[:-1]]) for target in target_list
    ]
    decoder_inputs = pad_sequence(decoder_input_list, batch_first=True, padding_value=PAD_IDX)

    # the targets are padded with PAD_IDX, and the provided training loop already
    # skips the pad positions in the loss, so we do not need to mask anything here
    decoder_targets = pad_sequence(target_list, batch_first=True, padding_value=PAD_IDX)

    # the very first token for generation, one BOS per example,
    # only used in evaluation
    initial_decoder_inputs = torch.full((len(batch), 1), BOS_ID, dtype=torch.long)

    # this order must match the unpacking in the provided train loop
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # on test the batch is only a list of question tensors, there are no targets
    encoder_ids = pad_sequence(batch, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.full((len(batch), 1), BOS_ID, dtype=torch.long)

    # only the three tensors that generation needs
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x
