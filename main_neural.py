from sentence_transformers import models, losses, evaluation, LoggingHandler, SentenceTransformer
from torch.utils.data import DataLoader
import logging
from transformers import logging as lg
import math
from neural_based import clu_evaluators
from utils import get_loss_fn, str2bool
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
import torch
from datetime import datetime
import argparse
import data_load

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="bi_encoder")#meta-llama/Llama-2-70b-chat-hf
parser.add_argument("--encoding_model", type=str, default="/home/chen_bowen/TextDedup/noise_output/mpnet_pre")#sentence-transformers/all-mpnet-base-v2
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--pooling", type=str2bool, default=False)
parser.add_argument("--num_epochs", type=int, default=8)
parser.add_argument("--warmup_epochs", type=int, default=8)
parser.add_argument("--eval_per_epoch", type=int, default=10)
parser.add_argument("--warm_up_perc", type=float, default=0.2)
parser.add_argument("--lr", type=float, default=2e-05)
parser.add_argument("--model_save_path", type=str, default="/home/chen_bowen/TextDedup/noise_output/scholar_noise_pretrain")
parser.add_argument("--loss_fn", type=str, default="contrastive")
parser.add_argument("--cuda", type=int, default=1)
parser.add_argument("--dataset", type=str, default="scholar")
args = parser.parse_args()
loss_params = {}
loss_params["margin"] = 0.2
loss_params["distance_metric"] = losses.SiameseDistanceMetric.COSINE_DISTANCE

device = "cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu"

lg.set_verbosity_error()
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

if args.dataset == "news_copy":
    dataset = data_load.load_news_copy("NEWS-COPY/")
elif args.dataset == "icdar":
    dataset = data_load.load_ICDAR("EngPostOCR/")
elif args.dataset == "noise_pretrain":
    dataset = data_load.noise_pretrain()
elif args.dataset == "scholar":
    dataset = data_load.load_scholar()
elif args.dataset == "train_all":
    news_dataset = data_load.load_news_copy("NEWS-COPY/")
    dataset = data_load.noise_pretrain()
if args.loss_fn in ["contrastive", "cosine"]:
    train_data = data_load.load_pair_news_copy_in_sent_transformer_format(dataset, set="train")
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
elif args.loss_fn == "triplet":
    train_data = data_load.load_triple_news_copy_in_sent_transformer_format(dataset, set="train")
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
elif args.loss_fn == "mnrl":
    train_data = data_load.load_triple_news_copy_in_sent_transformer_format(dataset, set="train")
elif args.loss_fn == "supcon":
    train_data = data_load.load_individual_news_copy_in_sent_transformer_format(dataset, set="train")
    tran_data_sampler = SentenceLabelDataset(train_data)
    train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
if args.dataset == "train_all":
    dev_data = data_load.load_pair_news_copy_in_sent_transformer_format(news_dataset, set="dev")
else:
    dev_data = data_load.load_pair_news_copy_in_sent_transformer_format(dataset, set="dev")
if args.method == "bi_encoder":
    if args.pooling:
        word_embedding_model = models.Transformer(args.encoding_model, max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
    else:
        model = SentenceTransformer(args.encoding_model, device=device)
    if args.dataset == "news_copy":
        evaluators = [
            evaluation.BinaryClassificationEvaluator.from_input_examples(dev_data),
            clu_evaluators.ClusterEvaluator.from_input_examples(dev_data, cluster_type="agglomerative")
        ]
    elif args.dataset == "icdar" or args.dataset == "noise_pretrain" or args.dataset=="scholar":
        evaluators = [
            evaluation.BinaryClassificationEvaluator.from_input_examples(dev_data),
        ]

    train_loss = get_loss_fn(model, loss_params, loss_fn=args.loss_fn)
    seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])
    logger.info("Evaluate model without neural")
    seq_evaluator(model, epoch=0, steps=0, output_path=args.model_save_path)
    warmup_steps = math.ceil(len(train_dataloader) * args.warmup_epochs)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=seq_evaluator,
        epochs=args.num_epochs,
        warmup_steps=math.ceil(len(train_dataloader) * args.warmup_epochs),
        output_path=args.model_save_path,
        evaluation_steps=112,
        checkpoint_save_steps=112,
        checkpoint_path=args.model_save_path,
        save_best_model=True,
        checkpoint_save_total_limit=10
    )
elif args.method == "cross_encoder":
    model = CrossEncoder(args.encoding_model, num_labels=1, device=device)
    if args.dataset == "news_copy":
        evaluators = [
            evaluation.BinaryClassificationEvaluator.from_input_examples(dev_data),
            clu_evaluators.ClusterEvaluator.from_input_examples(dev_data, cluster_type="agglomerative")
        ]
    elif args.dataset == "icdar" or args.dataset == "noise_pretrain":
        evaluators = [
            evaluation.BinaryClassificationEvaluator.from_input_examples(dev_data),
        ]
    seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])
    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * args.warm_up_perc)
    logger.info("Warmup-steps: {}".format(warmup_steps))
    # Train the model
    model.fit(train_dataloader=train_dataloader,
              evaluator=seq_evaluator,
              epochs=args.num_epochs,
              evaluation_steps=int(len(train_dataloader)*(1/args.eval_per_epoch)),
              loss_fct=torch.nn.BCEWithLogitsLoss(),
              optimizer_params={"lr": args.lr},
              warmup_steps=warmup_steps,
              output_path=f'output/{datetime.now().strftime("%Y-%m-%d_%H-%M")}')
