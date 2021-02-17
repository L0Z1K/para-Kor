import logging
import argparse
import pandas as pd 
import numpy as np

import torch
import pytorch_lightning as pl

from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader

class paraDataset(Dataset):
    def __init__(self, file_path, max_len=128):
        logging.info('[+] Init Data.')
        self.data = pd.read_csv(file_path)
        logging.info('[+] Load Data: Done.')
        self.tokenizer = BertTokenizerFast.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")

        
        self.max_len = max_len
        self.first = True
        #special_token_dict = {'additional_special_tokens':[self.bos_token, self.eos_token, self.q_token, self.a_token]}
        #self.tokenizer.add_special_tokens(special_token_dict)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        turn = self.data.iloc[idx]
        q = turn['A']
        a = turn['B']
        q_toked = [self.tokenizer.cls_token] + self.tokenizer.tokenize(q)
        q_len = len(q_toked)
        a_toked = [self.tokenizer.sep_token] + self.tokenizer.tokenize(a) + [self.tokenizer.sep_token]
        a_len = len(a_toked)

        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len/2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
                assert a_len > 0
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
        
        input_ids = q_toked+a_toked
        input_ids += ['[PAD]'] * (self.max_len - len(input_ids))
        labels = [self.tokenizer.mask_token] * q_len + a_toked[1:]
        labels += ['[PAD]'] * (self.max_len - len(labels))
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        if self.first:
            logging.info("contexts : {}".format(q))
            logging.info("toked ctx: {}".format(q_toked))
            logging.info("response : {}".format(a))
            logging.info("toked response : {}".format(a_toked))
            logging.info('labels {}'.format(labels))
            self.first = False
        return (self.tokenizer.convert_tokens_to_ids(input_ids),
                mask,
                self.tokenizer.convert_tokens_to_ids(labels))

class paraDataModule(pl.LightningDataModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.hparams = hparams
    
    @staticmethod
    def add_model_specific_args(parent_parser):
    # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        
        parser.add_argument('--batch-size',
                            type=int,
                            default=32,
                            help='batch size for training')

        parser.add_argument('--max-len',
                            type=int,
                            default=64,
                            help='max sentence length on input')

        parser.add_argument('--train_file',
                            type=str,
                            default="example.csv",
                            help="train file path")

        parser.add_argument('--num_workers',
                            type=int,
                            default=4,
                            help="number of workers")

        return parser
    
    def setup(self, stage):
        self.train = paraDataset(file_path=self.hparams.train_file,
                                 max_len=self.hparams.max_len)
    
    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        logging.info("[+] Load the train data.")
        train = DataLoader(dataset=self.train,
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            shuffle=True,
                            collate_fn=self._collate_fn,
                            pin_memory=True)
        return train