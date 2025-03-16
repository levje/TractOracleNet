#!/usr/bin/env python
import argparse

from argparse import RawTextHelpFormatter
from os.path import join
import comet_ml

from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from TractOracleNet.models.transformer import TransformerOracle
from TractOracleNet.trainers.data_module import StreamlineDataModule
import torch

# Set the default precision to float32 to
# speed up training and reduce memory usage
torch.set_float32_matmul_precision("medium")


class TractOracleNetTransformerTesting():
    """ Train a Transformer model to score streamlines.
    """

    def __init__(
        self,
        train_dto: dict,
    ):
        """ Initialize the training process.
        """
        self.checkpoint = train_dto['checkpoint']

        # Data loading parameters
        self.num_workers = train_dto['num_workers']
        self.batch_size = train_dto['batch_size']
        self.n_layers = train_dto['n_layers']
        self.n_head = train_dto['n_head']

        # Data files
        self.dataset_file = train_dto['dataset_file']

    def test(
        self,
    ):
        """ Train the model.
        """
        # Get example input to define NN input size
        # 128 points directions -> 127 3D directions
        self.input_size = (128-1) * 3  # Get this from datamodule ?
        self.output_size = 1

        if self.checkpoint:
            model = TransformerOracle.load_from_checkpoint(self.checkpoint)
        else:
            model = TransformerOracle(
                self.input_size, self.output_size, self.n_head,
                self.n_layers, 0)

        # Instanciate the datamodule
        dm = StreamlineDataModule(self.dataset_file, self.batch_size, self.num_workers)

        # Training
        comet_logger = CometLogger(
            save_dir=".",
            project_name="tractoracletesting")

        # Log parameters
        comet_logger.log_hyperparams({
            "model": TransformerOracle.__name__,
            "n_layers": self.n_layers,
            "n_head": self.n_head,
            "batch_size": self.batch_size})

        # Log the learning rate during training as it will vary
        # from Cosine Annealing
        lr_monitor = LearningRateMonitor(logging_interval='step')

        # Define the trainer
        # Mixed precision is used to speed up training and
        # reduce memory usage
        trainer = Trainer(logger=comet_logger,
                          log_every_n_steps=1,
                          num_sanity_val_steps=0,
                          max_epochs=1,
                          enable_checkpointing=True,
                          default_root_dir="",
                          precision='16-mixed',
                          callbacks=[lr_monitor])
        # Test the model
        dm.setup("fit")
        trainer.test(model, dm)


def add_args(parser):
    parser.add_argument('dataset_file', type=str,
                        help='Training dataset.')
    parser.add_argument('checkpoint', type=str,
                        help='Path to checkpoint. If not provided, '
                             'train from scratch.')
    parser.add_argument('--batch_size', type=int, default=(2**11+768),
                        help='Batch size, in number of streamlines.')
    parser.add_argument('--num_workers', type=int, default=20,
                        help='Number of workers for dataloader.')
    parser.add_argument('--n_head', type=int, default=4,
                        help='Number of attention heads.')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of encoder layers.')

def parse_args():
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)
    add_args(parser)
    args = parser.parse_args()
    return args


def main():
    " Main function."

    args = parse_args()
    # Train the model
    training = TractOracleNetTransformerTesting(vars(args))
    training.test()


if __name__ == "__main__":
    main()
