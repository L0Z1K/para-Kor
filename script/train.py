import argparse
import logging
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model import paraKor
from dataloader import paraDataModule
from utils import CheckpointEveryNSteps

parser = argparse.ArgumentParser(description='Paraphrasing Model on KoGPT-3')

parser.add_argument('--test',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

parser.add_argument('--model_params',
                    type=str,
                    default='',
                    help='model binary for starting chat')

parser.add_argument('--train',
                    action='store_true',
                    default=False,
                    help='for training')

parser.add_argument('--load',
                    action='store_true',
                    default=False,
                    help='training model from checkpoint')

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

parser = paraKor.add_model_specific_args(parser)
parser = paraDataModule.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()
logging.info(args)

if __name__ == "__main__":
    if args.train:
        if args.load:
            model = paraKor.load_from_checkpoint(args.model_params)
        else:
            model = paraKor(args)
        model.train()
        dm = paraDataModule(args)
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=[CheckpointEveryNSteps(20000)], gradient_clip_val=1.0)
        trainer.fit(model, dm)
    if args.test:
        model = paraKor.load_from_checkpoint(args.model_params)
        model.test()