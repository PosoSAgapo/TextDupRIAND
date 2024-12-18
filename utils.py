import re
import logging

from sentence_transformers import losses
from loss_func import SupConLoss

# import cugraph as cnx
# import cudf as gd
#load dataset into huggingface dataset format
#1 for same news, 0 for different news

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.
    """
    prefix_re = re.compile(fr'^(?:{"|".join(prefices)})')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

def embedding(dataset, model):
    corpus_embeddings = model.encode(dataset["text1"])
    return corpus_embeddings

def str2bool(v):
    if v == "True":
        return True
    elif v == "False":
        return False

def get_loss_fn(model, loss_params,loss_fn="constrative"):
    if loss_fn == "contrastive":
        train_loss = losses.OnlineContrastiveLoss(
            model=model,
            distance_metric=loss_params['distance_metric'],
            margin=loss_params['margin']
        )
    elif loss_fn == "cosine":
        train_loss = losses.CosineSimilarityLoss(model=model)
    elif loss_fn == "triplet":
        train_loss = losses.TripletLoss(
            model=model,
            distance_metric=loss_params['distance_metric'],
            triplet_margin=loss_params['margin']
        )
    elif loss_fn == "mnrl":
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    elif loss_fn == "supcon":
        train_loss = SupConLoss(model=model)
    return train_loss



