from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import os
import pytorch_lightning as pl
import sys 
sys.path.append('/app/')
from pitchclass2vec import encoding, model
from tasks.segmentation.data import BillboardDataset, SegmentationDataModule
from tasks.segmentation.deeplearning_models.lstm import LSTMBaselineModel
from tasks.segmentation.deeplearning_models.transformer import TransformerModel
import pitchclass2vec.model as model
import pitchclass2vec.encoding as encoding
from pitchclass2vec.pitchclass2vec import NaiveEmbeddingModel

from evaluate import load_pitchclass2vec_model
import argparse
import wandb 
import logging
import wandb
from pathlib import Path
from distutils.util import strtobool
logging.disable(logging.CRITICAL)
RANDOM_SEED = 42
pl.seed_everything(seed=RANDOM_SEED)


ENCODING_MAP = {
    "root-interval": encoding.RootIntervalDataset, # return source, target, y
    "timed-root-interval": encoding.TimedRootIntervalDataset, # return source, target, y, duration
    "chord2vec": encoding.Chord2vecDataset,
}



def train(exp_args, segmentation_train_args):
    pl.seed_everything(seed=segmentation_train_args.get("seed", 42), workers=True)

    # Create a folder to save the segmentation model if it's not exist
    segmentation_out = segmentation_train_args.get("out")
    if not os.path.exists(segmentation_out): os.makedirs(segmentation_out)

    # Use advance embedding model to convert chord string into vector

    encoder = ENCODING_MAP[exp_args.get("encoder")]
    embedding_model = NaiveEmbeddingModel(
                            encoding_model=encoder, 
                            embedding_dim=3, # dim=3 because each '24 basic chords' only contain 3 notes
                            norm=False) 



    # TODO: At the moment its train on Billboard dataset, will be train on STEM dataset in the future
    # Prepare dataset for Segmentation model trainning by Billboard Dataset
    data = SegmentationDataModule(  dataset_cls=BillboardDataset, 
                                    embedding_model=embedding_model, 
                                    batch_size = segmentation_train_args.get("batch_size",256), 
                                    test_mode = segmentation_train_args.get("test_mode", True),
                                    full_chord = segmentation_train_args.get("full_chord", False)
                                    )

    # Prepare Model
    # If we not using pitchclass2vec_model then embedding_dim must be 3
    embedding_dim = 3

    transformer_model = TransformerModel(segmentation_train_args)


    # Set up Weight&Bias for monitering the trainning process
    if not segmentation_train_args.get("disable_wandb", False):
        wandb.init(
            # Set the project where this run will be logged
            project="Segmentation_with_Transformer", 
            name=f"{ segmentation_train_args.get('wandb_run_name', 'None') }",
            # Track hyperparameters and run metadata
            # TODO: need to update the config here
            config={
                # Add any other parameters you want to track
                "num_classes": segmentation_train_args["num_classes"],
                # TODO: update this part base on segmentation_train_args
                # "embedding_dim": embedding_dim,
                # "num_layers": segmentation_train_args["num_layers"],
                # "dropout": segmentation_train_args["dropout"],
                # "learning_rate": segmentation_train_args["learning_rate"],
                # "batch_size": segmentation_train_args["batch_size"],
                # "max_epochs": segmentation_train_args["max_epochs"],
                # "factor": segmentation_train_args["factor"],
                # "patience": segmentation_train_args["patience"]
            }
        )
        wandb.watch(transformer_model)

    # TODO: monitor acc as well
    callbacks = [
        pl.callbacks.ModelCheckpoint(save_top_k=1,
                                    monitor="train/loss",
                                    mode="min",
                                    dirpath=segmentation_train_args.get("out"),
                                    filename=segmentation_train_args.get('wandb_run_name'),
                                    every_n_epochs=1)
    ] 

    trainer = pl.Trainer(   max_epochs=segmentation_train_args.get("max_epochs"), 
                            accelerator="auto", 
                            devices=1,
                            enable_progress_bar=True,
                            callbacks=callbacks)


    trainer.fit(transformer_model, data)

    wandb.save(str(Path(segmentation_train_args.get("out")) / f"{segmentation_train_args.get('wandb_run_name')}"))

    test_metrics = trainer.test(transformer_model, data)
    # Use pd.concat instead of pd.append
    new_row_df = pd.DataFrame([{
        "encoding": exp_args.get("encoder"),  **test_metrics[0]
    }])

    experiments_df = pd.DataFrame(columns=[
    "encoding", "model", "path", "test_p_precision", "test_p_recall",  "test_p_f1",  "test_under",  "test_over",  "test_under_over_f1"
    ])

    experiments_df = pd.concat([experiments_df, new_row_df], ignore_index=True)
    
    return experiments_df


def str2bool(v):
    return bool(strtobool(v))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Segmentation Model by Transformer.")

    # Arguments for exp_args
    parser.add_argument("--encoder", type=str, default="root-interval",
                        choices=["root-interval", "other_choice"],  # Update with actual choices
                        help="Type of encoder to use.")

    # Arguments for segmentation_train_args
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    parser.add_argument("--test_mode", action='store_true', help="Whether to use test mode.")
    parser.add_argument("--full_chord", action='store_true', help="Whether to use full chords.")
    parser.add_argument("--disable_wandb", action='store_true', help="Whether to disable wandb.")
    parser.add_argument("--wandb_run_name", type=str, default="transformer_test_run", help="The run name for Weights & Biases tracking.")
    parser.add_argument("--out", type=str, default="/app/segmentation_out", help="Output path for saving the model checkpoints.")

    parser.add_argument("--input_dim", type=int, default=3, help="Hidden dimensionality of the input.")
    parser.add_argument("--model_dim", type=int, default=128, help="Hidden dimensionality to use inside the Transformer.")
    parser.add_argument("--feedforward_dim", type=int, default=256, help="Dimensionality of the feedforward network model.")
    parser.add_argument("--num_classes", type=int, default=14, help="Number of classes to predict per sequence element.")
    parser.add_argument("--num_heads", type=int, default=3, help="Number of heads to use in the Multi-Head Attention blocks.")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of encoder and decoder blocks to use.")
    parser.add_argument("--decoder_max_length", type=int, default=500, help="Maximum length of the decoder.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--warmup", type=int, default=100, help="Number of warmup steps for learning rate scheduler.")
    parser.add_argument("--max_iters", type=int, default=1, help="Number of maximum iterations the model is trained for.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate to apply inside the model.")
    parser.add_argument("--input_dropout", type=float, default=0.1, help="Dropout rate to apply on the input features.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--max_epochs", type=int, default=1, help="Maximum number of epochs to train for.")
    parser.add_argument("--factor", type=float, default=0.5, help="Factor by which the learning rate will be reduced.")
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs with no improvement after which learning rate will be reduced.")
    parser.add_argument("--init_method", type=str, default="xavier", choices=["xavier", "orthogonal"], help="Method for initializing the weights.")

    args = parser.parse_args()
        


    args = parser.parse_args()
    # Combine the parsed arguments into the expected structure for train function
    exp_args = {
        "encoder": args.encoder
    }

    # Include all segmentation_train_args that you need
    segmentation_train_args = {
        "seed": args.seed,
        "test_mode": args.test_mode,
        "full_chord": args.full_chord,
        "disable_wandb": args.disable_wandb,
        "wandb_run_name": args.wandb_run_name,
        "out": args.out,

        "input_dim": args.input_dim,
        "model_dim": args.model_dim,
        "feedforward_dim": args.feedforward_dim,
        "num_classes": args.num_classes,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "decoder_max_length": args.decoder_max_length,
        "device": args.device,
        "lr": args.lr,
        "warmup": args.warmup,
        "max_iters": args.max_iters,
        "dropout": args.dropout,
        "input_dropout": args.input_dropout,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "factor": args.factor,
        "patience": args.patience,
        "init_method": args.init_method
    }

    experiments_df = train(exp_args, segmentation_train_args)
    print(experiments_df)
