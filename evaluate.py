import argparse

import torch
import numpy as np
import pytorch_lightning as pl
import json

from choco.corpus import ChoCoHarteAnnotationsCorpus
from pitchclass2vec.data import ChocoDataModule
import pitchclass2vec.encoding as encoding
import pitchclass2vec.model as model
from pitchclass2vec.pitchclass2vec import Pitchclass2VecModel
from gensim_evaluations.methods import OddOneOut
from train import MODEL_MAP, ENCODING_MAP
from gensim.models import KeyedVectors

def load_pitchclass2vec_model(encoding: str, model: str, path: str):
    if encoding == "text":
        model = KeyedVectors.load(path).wv

        # FIXME: Workaround to use FastText from gensim with the current OddOneOut
        # implementation
        if "FastText" in str(model):
            model.has_index_for = lambda _: True
    else:
        model = Pitchclass2VecModel(ENCODING_MAP[encoding], 
                                    MODEL_MAP[model],
                                    path)
    return model

def evaluate(encoding: str, model: str, path: str, config: str):
    model = load_pitchclass2vec_model(encoding, model, path)
    with open(config) as f:
        config = json.load(f)

    metrics = {}
    metrics["odd_one_out"] = OddOneOut(
        { k: v for k, v in config.items() if k != "vocab" },
        model, 
        allow_oov=True,
        vocab=config["vocab"],
        k_in=4
    )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pitchclass2vec embedding.")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--encoding", 
                        choices=list(ENCODING_MAP.keys()), 
                        required=True, 
                        default="root-interval")
    parser.add_argument("--model", 
                        choices=list(MODEL_MAP.keys()), 
                        required=True, 
                        default="fasttext")
    
    args = parser.parse_args()
    
    evaluation = evaluate(args.encoding, args.model, args.path, args.config)
    for metric, metric_eval in evaluation.items():
        print(f"{metric}:")
        accuracy, accuracy_per_cat, _, _, _ = metric_eval
        print(f"Accuracy: {accuracy}")
        for cat, acc in accuracy_per_cat.items():
            print(f"\ton {cat}: {acc}")