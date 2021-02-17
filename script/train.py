import argparse
import logging
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model import paraKor
from dataloader import paraDataModule

class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{epoch=}_{global_step=}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

parser = argparse.ArgumentParser(description='Paraphrasing Model on KoGPT-3')

parser.add_argument('--chat',
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
    if args.chat:
        model = paraKor.load_from_checkpoint(args.model_params)
        model.chat()