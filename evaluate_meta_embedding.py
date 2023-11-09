import argparse

import torch
import numpy as np
import pytorch_lightning as pl
import json

from more_itertools import unique_everseen, powerset

from choco.corpus import ChoCoHarteAnnotationsCorpus
from pitchclass2vec.data import ChocoDataModule
import pitchclass2vec.encoding as encoding
import pitchclass2vec.model as model
from pitchclass2vec.pitchclass2vec import Pitchclass2VecModel, MetaPitchclass2VecModel
from gensim_evaluations.methods import odd_one_out
from tasks.train_embedding_model.embedding_train import MODEL_MAP, ENCODING_MAP
from gensim.models import KeyedVectors
from evaluate import load_pitchclass2vec_model

def evaluate(models, method, config: str):
    embedding_models = [load_pitchclass2vec_model(*m) for m in models ]
    model = MetaPitchclass2VecModel(embedding_models)

    with open(config) as f:
        config = json.load(f)

    metrics = {}
    metrics["odd_one_out"] = odd_one_out(
        { k: v for k, v in config.items() if k != "vocab" },
        model, 
        allow_oov=True,
        vocab=config["vocab"],
        k_in=4
    )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pitchclass2vec meta-embedding.")
    parser.add_argument("-e", "--embeddings", nargs="+", action="append")
    parser.add_argument("--method", type=str, required=False, default="concat")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--product", action="store_true")
    
    args = parser.parse_args()
    
    embeddings = list(map(tuple, args.embeddings))
    if args.product:
        embeddings = list(filter(lambda x: len(x) > 1, powerset(embeddings)))
    else:
        embeddings = [embeddings]
    
    for emb in embeddings:
        print("-" * 15)
        print(emb)
        print("-" * 15)
        evaluation = evaluate(emb, args.method, args.config)
        for metric, metric_eval in evaluation.items():
            print(f"{metric}:")
            accuracy, accuracy_per_cat, _, _, _ = metric_eval
            print(f"Accuracy: {accuracy}")
            for cat, acc in accuracy_per_cat.items():
                print(f"\ton {cat}: {acc}")