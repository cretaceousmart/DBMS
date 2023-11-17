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

EMBEDDING_MODEL_MAP = {
    "word2vec": model.Word2vecModel,
    "fasttext": model.FasttextModel,
    "scaled-loss-fasttext": model.ScaledLossFasttextModel,
    "emb-weighted-fasttext": model.EmbeddingWeightedFasttextModel,
    "rnn-weighted-fasttext": model.RNNWeightedFasttextModel,
}

def train(exp_args, segmentation_train_args):
    pl.seed_everything(seed=segmentation_train_args.get("seed", 42), workers=True)

    # Create a folder to save the segmentation model if it's not exist
    segmentation_out = segmentation_train_args.get("out")
    if not os.path.exists(segmentation_out): os.makedirs(segmentation_out)

    # Use advance embedding model to convert chord string into vector
    if segmentation_train_args.get("use_pitchclass2vec_model"):
    # load encoder and embedding model
        encoder, embedding_model = exp_args.get("encoder"), exp_args.get("embedding_model")
        embedding_model_path = exp_args.get("embedding_model_path")
        embedding_model = load_pitchclass2vec_model(encoder, embedding_model, embedding_model_path)
    
    else:
        print("Jie Log: Use root-interval as enbedding!!!")
        encoder = ENCODING_MAP[exp_args.get("encoder")]
        embedding_model = NaiveEmbeddingModel(
                                encoding_model=encoder, 
                                embedding_dim=3, # dim=3 because each '24 basic chords' only contain 3 notes
                                norm=segmentation_train_args.get("norm_in_embedding",False)) 


    # Prepare dataset for Segmentation model trainning by Billboard Dataset
    data = SegmentationDataModule(  dataset_cls=BillboardDataset, 
                                    embedding_model=embedding_model, 
                                    batch_size = segmentation_train_args.get("batch_size",256), 
                                    test_mode = segmentation_train_args.get("test_mode", True),
                                    full_chord = segmentation_train_args.get("full_chord", False)
                                    )

    # Prepare Model
    # If we not using pitchclass2vec_model then embedding_dim must be 3
    embedding_dim = embedding_model.vector_size if segmentation_train_args.get("use_pitchclass2vec_model") else 3

    lstm_model = LSTMBaselineModel(
        segmentation_train_args = segmentation_train_args,
        num_labels=segmentation_train_args["num_labels"],
        embedding_dim=embedding_dim,
        hidden_size=segmentation_train_args["hidden_size"],
        num_layers=segmentation_train_args["num_layers"],
        dropout=segmentation_train_args["dropout"],
        learning_rate=segmentation_train_args["learning_rate"],
    )

    transformer_model = TransformerModel(segmentation_train_args)

    # Set up Weight&Bias for monitering the trainning process
    if not segmentation_train_args.get("disable_wandb", False):
        wandb.init(
            # Set the project where this run will be logged
            project="pitchclass2vec_Segmentation", 
            name=f"{ segmentation_train_args.get('wandb_run_name', 'None') }",
            # Track hyperparameters and run metadata
            config={
                # Add any other parameters you want to track
                "num_labels": segmentation_train_args["num_labels"],
                "embedding_dim": embedding_dim,
                "hidden_size": segmentation_train_args["hidden_size"],
                "num_layers": segmentation_train_args["num_layers"],
                "dropout": segmentation_train_args["dropout"],
                "learning_rate": segmentation_train_args["learning_rate"],
                "batch_size": segmentation_train_args["batch_size"],
                "max_epochs": segmentation_train_args["max_epochs"],
                "factor": segmentation_train_args["factor"],
                "patience": segmentation_train_args["patience"]
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

    if segmentation_train_args.get("model_type", False) == "lstm":
        trainer.fit(lstm_model, data)
    else:
        trainer.fit(transformer_model, data)

    wandb.save(str(Path(segmentation_train_args.get("out")) / f"{segmentation_train_args.get('wandb_run_name')}"))

    test_metrics = trainer.test(transformer_model, data)
    # Use pd.concat instead of pd.append
    new_row_df = pd.DataFrame([{
        "encoding": exp_args.get("encoder"), "embedding_model": exp_args.get("embedding_model"), "embedding_model_path": exp_args.get("embedding_model_path"), **test_metrics[0]
    }])

    experiments_df = pd.DataFrame(columns=[
    "encoding", "model", "path", "test_p_precision", "test_p_recall",  "test_p_f1",  "test_under",  "test_over",  "test_under_over_f1"
    ])

    experiments_df = pd.concat([experiments_df, new_row_df], ignore_index=True)
    
    return experiments_df


def str2bool(v):
    return bool(strtobool(v))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Segmentation Model.")

    # Arguments for exp_args
    parser.add_argument("--encoder", type=str, default="root-interval",
                        choices=list(ENCODING_MAP.keys()), 
                        help="Type of encoder to use.")
    parser.add_argument("--embedding_model", type=str, default="fasttext",
                        choices=list(EMBEDDING_MODEL_MAP.keys()), 
                        help="Type of embedding model to use.")
    parser.add_argument("--embedding_model_path", type=str, required=True,
                        help="Path to the embedding model checkpoint.")

    # Arguments for segmentation_train_args
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")

    parser.add_argument("--test_mode", type=str2bool, default=False, nargs='?', const=True, help="Whether to use test mode (default: %(default)s).")
    
    parser.add_argument("--full_chord", type=str2bool, default=False, nargs='?', const=True, help="Whether to use full chords (default: %(default)s).")

    parser.add_argument("--use_pitchclass2vec_model", type=str2bool, default=False, nargs='?', const=True, help="Whether to use pitchclass2vec model (default: %(default)s).")

    parser.add_argument("--norm_in_embedding", type=str2bool, default=False, nargs='?', const=True, help="Whether to use norm during embedding (default: %(default)s).")


    parser.add_argument("--disable_wandb", type=str2bool, default=False, nargs='?', const=True,
                        help="Whether to disable wandb (default: %(default)s).")
    parser.add_argument("--num_labels", type=int, default=11, help="The number of labels.")
    
    parser.add_argument("--hidden_size", type=int, default=100, help="The number of features in the hidden state of the LSTM.")

    parser.add_argument("--num_layers", type=int, default=5, help="Number of recurrent layers.")
    
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")

    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")

    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    
    parser.add_argument("--max_epochs", type=int, default=150, help="Maximum number of epochs to train for.")

    parser.add_argument("--factor", type=float, default=0.5, help="Factor by which the learning rate will be reduced.")
    
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs with no improvement after which learning rate will be reduced.")

    parser.add_argument("--wandb_run_name", type=str, default="No_name.ckpt", help="The run name for Weights & Biases tracking.")
    
    parser.add_argument("--out", type=str, default="/app/segmentation_out", help="Output path for saving the model checkpoints.")

    parser.add_argument("--init_method", type=str, default="xavier", help="Method for initialize the weights.")

    args = parser.parse_args()
    # Combine the parsed arguments into the expected structure for train function
    exp_args = {
        "encoder": args.encoder,
        "embedding_model": args.embedding_model,
        "embedding_model_path": args.embedding_model_path
    }

    # Include all segmentation_train_args that you need
    segmentation_train_args = {
        "seed": args.seed,
        "test_mode": args.test_mode,
        "full_chord": args.full_chord,
        "disable_wandb": args.disable_wandb,
        "num_labels": args.num_labels,
        # "embedding_dim": args.embedding_dim,  # This will be None if not provided TODO: figure out what is this
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "factor": args.factor,
        "patience": args.patience,
        "wandb_run_name": args.wandb_run_name,
        "out": args.out,
    }

    experiments_df = train(exp_args, segmentation_train_args)
    print(experiments_df)
