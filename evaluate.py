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
# from tasks.train_embedding_model.embedding_train import MODEL_MAP, ENCODING_MAP
from gensim.models import KeyedVectors

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

def load_pitchclass2vec_model(encoding: str, model: str, path: str):
    if encoding == "text":
        model = KeyedVectors.load(path).wv

        # FIXME: Workaround to use FastText from gensim with the current OddOneOut
        # implementation
        if "FastText" in str(model):
            model.has_index_for = lambda _: True
    else:
        model = Pitchclass2VecModel(ENCODING_MAP[encoding], 
                                    EMBEDDING_MODEL_MAP[model],
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