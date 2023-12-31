import argparse
import os
import json
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
import wandb

from choco.corpus import ChoCoHarteAnnotationsCorpus
from pitchclass2vec.data import ChocoDataModule
import pitchclass2vec.encoding as encoding
import pitchclass2vec.model as model

from gensim.models import Word2Vec, FastText


ENCODING_MAP = {
    "root-interval": encoding.RootIntervalDataset, # return source, target, y
    "timed-root-interval": encoding.TimedRootIntervalDataset, # return source, target, y, duration
    # "all-interval": encoding.AllIntervalDataset,
    # "timed-all-interval": encoding.TimedAllIntervalDataset,
    "chord2vec": encoding.Chord2vecDataset,

    # The below encoding model may not useful for us
    # "text": encoding.HarteTextDataset,
    # "rdf": encoding.RDFChordsDataset,
}



# Here Model means Embeded model, but not the DL model (LSTM use as the base model in segmentation.py)
# TODO: Figure out which model is pitchesclass2vec (the best model)
MODEL_MAP = {
    "word2vec": model.Word2vecModel,
    "fasttext": model.FasttextModel,
    "scaled-loss-fasttext": model.ScaledLossFasttextModel,
    "emb-weighted-fasttext": model.EmbeddingWeightedFasttextModel,
    "rnn-weighted-fasttext": model.RNNWeightedFasttextModel,

    # The below embedding model may not useful for us
    # "randomwalk-rdf2vec": model.RandomWalkRdf2VecModel,
    # "wlwalk-rdf2vec": model.WLWalkRdf2VecModel,
    # "halkwalk-rdf2vec": model.HALKWalkRdf2VecModel,
}


def train_with_gensim(choco, encoding, model, out, **kwargs):
    dataset_cls = ENCODING_MAP[encoding]
    data = ChocoDataModule(
        choco,
        dataset_cls,
        batch_size=kwargs.get("batch_size", 1024),
        context_size=kwargs.get("context", 5),
        negative_sampling_k=kwargs.get("negative_sampling_k", 20))
        
    if encoding == "text":
        data.prepare_data()
        data = data.dataset

        if model == "word2vec":
            model = Word2Vec(sentences=data, 
                vector_size=kwargs.get("embedding_dim", 100), 
                window=kwargs.get("context", 5),
                hs=0,
                negative=kwargs.get("negative_sampling_k", 20),
                sg=1,
                seed=kwargs.get("seed", 42),
                epochs=kwargs.get("max_epochs", 42))
            model.save(str(Path(out) / "model.ckpt"))
        elif model == "fasttext":
            model = FastText(sentences=data, 
                vector_size=kwargs.get("embedding_dim", 100), 
                window=kwargs.get("context", 5),
                negative=kwargs.get("negative_sampling_k", 20),
                hs=0,
                sg=1,
                seed=kwargs.get("seed", 42),
                epochs=kwargs.get("max_epochs", 42))
            model.save(str(Path(out) / "model.ckpt"))

def train_with_rdf2vec(choco, encoding, model, out, **kwargs):
    data = ENCODING_MAP[encoding](choco)
    model = MODEL_MAP[model](**kwargs)
    model.train(data)
    model.save(str(Path(out) / "model.ckpt"))


def train_with_torch(choco, encoding, model, out, **kwargs):
    # Initilize Weight&Bias
    if not kwargs.get("disable_wandb", False):

        wandb.init(
            # Set the project where this run will be logged
            project="pitchclass2vec", 
            name=f"{ kwargs.get('wandb_run_name', 'None') }",
            # Track hyperparameters and run metadata
            config={
                "encoding": encoding,
                "embedding_model": model,
                "batch_size": kwargs.get("batch_size",1024),
                "context_size": kwargs.get("context_size",5),
                "negative_sampling_k": kwargs.get("negative_sampling_k",20),
            }
        )
        
        # Config Weight&Bias
        wandb.config.update({
            "encoding": encoding,
            "model": model,
            **kwargs
        })


    # Data Preparation：
    dataset_cls = ENCODING_MAP[encoding]
    data = ChocoDataModule(
        choco,
        dataset_cls,
        batch_size=kwargs.get("batch_size", 1024),
        context_size=kwargs.get("context", 5),
        negative_sampling_k=kwargs.get("negative_sampling_k", 20))

    # Model Initialization：
    model_cls = MODEL_MAP[model]
    model = model_cls(**kwargs)

    # Use callbacks of PyTorch Lightning
    # Check the loss for each epoch and save the model in /out, only save the best model (min loss)
    file_name = f"{kwargs.get('wandb_run_name', '')}"
    callbacks = [
        pl.callbacks.ModelCheckpoint(save_top_k=1,
                                    monitor="train/loss",
                                    mode="min",
                                    dirpath=out,
                                    filename=file_name,
                                    every_n_epochs=1)
    ]


    # Early stopping
    if kwargs.get("early_stop_patience", -1) != -1:
        callbacks.append(pl.callbacks.EarlyStopping(
            monitor="train/loss",
            min_delta=0.00,
            patience=kwargs.get("early_stop_patience", 2)))


    # Build trainer
    trainer = pl.Trainer(max_epochs=kwargs.get("max_epochs", 5),
                        accelerator="auto",
                        # logger=None if kwargs.get("disable_wandb", False) else logger,
                        logger=None,
                        devices=1,
                        callbacks=callbacks)

    trainer.fit(model, datamodule=data)

    wandb.save(str(Path(out) / f"{file_name}.ckpt"))
        



def train(choco, encoding, model, out, **kwargs):
    pl.seed_everything(seed=kwargs.get("seed", 42), workers=True)
    
    # save training metadata
    if not os.path.exists(out):
        os.makedirs(out)

    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({
            "encoding": encoding,
            "model": model,
            **kwargs
        }, f)

    if encoding == "rdf":
        train_with_rdf2vec(choco, encoding, model, out, **kwargs)
    elif encoding == "text":
        train_with_gensim(choco, encoding, model, out, **kwargs)
    else:
        train_with_torch(choco, encoding, model, out, **kwargs)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pitchclass2vec embedding.")
    parser.add_argument("--choco", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--encoding", 
                    choices=list(ENCODING_MAP.keys()), 
                        default="root-interval")
    parser.add_argument("--model", 
                        choices=list(MODEL_MAP.keys()), 
                        default="fasttext")

    
    args, unknown = parser.parse_known_args()
    unknown = {
        key.replace("--", "").replace("-", "_"): int(val) if val.replace("-", "").isdigit() else val
        for key, val in zip(unknown[::2], unknown[1::2])
    }

    train(args.choco, args.encoding, args.model, args.out, **unknown)
