import argparse
import logging

import torch
import torch.nn as nn
import pytorch_lightning as pl

from transformers import BertTokenizerFast, GPT2LMHeadModel
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

class paraKor(pl.LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.hparams = hparams
        self.kogpt3 = GPT2LMHeadModel.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")
        self.loss_function = nn.CrossEntropyLoss(reduction='none')
        self.neg = -1e18
        self.tokenizer = BertTokenizerFast.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')

        return parser

    def forward(self, inputs):
        output = self.kogpt3(inputs)[0]
        return output

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        tensorboard_logs = {'train_loss': loss_avg}
        return {'loss': loss_avg, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader().dataset) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]
    
    def test(self):
        with torch.no_grad():
            while 1:
                q = input('A: ').strip()
                if q == 'quit':
                    break
                q_tok = [self.tokenizer.cls_token] + self.tokenizer.tokenize(q) + [self.tokenizer.sep_token]

                input_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(q_tok)).unsqueeze(dim=0)
                gen = self.kogpt3.generate(input_ids,
                                           num_beams=5,
                                           max_length=self.hparams.max_len,
                                           no_repeat_ngram_size=2,
                                           pad_token_id=3) # eos_token_id = 3
                
                gen = self.tokenizer.convert_ids_to_tokens(gen[0].tolist())
                gen = gen[gen.index('[SEP]')+1:]
                try:
                    gen = gen[:gen.index('[SEP]')]
                except ValueError:
                    pass

                answer = self.tokenizer.convert_tokens_to_string(gen)
                print(f"B: {answer}")