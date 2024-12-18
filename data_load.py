import random
import spacy
import datasets
from sentence_transformers.readers import InputExample
import os
from cluster_utils import clusters_from_edges
import re
from tqdm import tqdm
import copy
import textdistance
import noise_functions
import pickle
import pandas as pd

random.seed(77774)
def load_news_copy(base_addr):
    dataset = datasets.load_dataset("csv", data_files={"train": base_addr + "training_sets/train_set.csv",
                                                       "dev": base_addr + "training_sets/dev_set.csv"},
                                    sep="\t")
    dataset = dataset.remove_columns(["Unnamed: 0"])
    dataset = dataset.rename_column("Text 1", "text1")
    dataset = dataset.rename_column("Text 2", "text2")
    dataset = dataset.map(lambda x: {"label": 1 if x["Label"] == "same" else 0}, remove_columns=["Label"])
    return dataset
# def load_news_copy_in_hug_dataset_format(base_addr, tokenizer):
#     tokenized_dataset = dataset.map(lambda examples: tokenizer(examples["text1"], examples["text2"], return_tensors="np"), batched=True)
#     return dataset, tokenized_dataset

def load_pair_news_copy_in_sent_transformer_format(dataset, set="train"):
    train_set = [dataset[set]['text1'], dataset[set]['text2'], dataset[set]["label"]]
    paired_data = []
    for i in range(len(train_set[0])):
        paired_data.append(InputExample(texts=[train_set[0][i], train_set[1][i]], label=float(train_set[2][i])))
    print(f'{len(paired_data)} {type} pairs')
    return paired_data

def load_triple_news_copy_in_sent_transformer_format(dataset, set="train"):
    sentence_1_list = dataset[set]['text1']
    sentence_2_list = dataset[set]['text2']
    labels = dataset[set]['label']
    def add_to_samples(sent1, sent2, label):
        if sent1 not in anchor_dict:
            anchor_dict[sent1] = {'same': set(), 'different': set()}
        anchor_dict[sent1][label].add(sent2)
    anchor_dict = {}
    for i in range(len(sentence_1_list)):
        add_to_samples(sentence_1_list[i], sentence_2_list[i], labels[i])
        add_to_samples(sentence_2_list[i], sentence_1_list[i], labels[i])

    triplet_data = []
    for anchor, others in anchor_dict.items():
        while len(others['same']) > 0 and len(others['different']) > 0:
            same_sent = random.choice(list(others['same']))
            dif_sent = random.choice(list(others['different']))

            triplet_data.append(InputExample(texts=[anchor, same_sent, dif_sent]))

            others['same'].remove(same_sent)
            others['different'].remove(dif_sent)

    print(f'{len(triplet_data)} {type} triplets')

    return triplet_data

def load_individual_news_copy_in_sent_transformer_format(dataset, set="train"):
    listed_set = [dataset[set]['text1'], dataset[set]['text2'], dataset[set]["label"]]
    edges_list = []
    for i in range(len(listed_set[0])):
        if listed_set[2][i] == 1:
            edges_list.append([listed_set[0][i], listed_set[1][i]])
    cluster_dict = clusters_from_edges(edges_list)
    indv_data = []
    guid = 1
    for cluster_id in list(cluster_dict.keys()):
        for text in cluster_dict[cluster_id]:
            indv_data.append(InputExample(guid=guid, texts=[text], label=cluster_id))
            guid += 1
    print(f"{len(indv_data)} {type} examples")
    return indv_data

def remove_alignment_errors(raw, aligned_raw, gold_aligned):
    processed_raw = []
    processed_gold = []
    raw = raw.replace("[OCR_toInput] ","")
    aligned_raw = aligned_raw.replace("[OCR_aligned] ","")
    gold_aligned = gold_aligned.replace("[ GS_aligned] ","")
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    start_idx = 0
    for idx, (align_raw_char, gold_raw_char) in enumerate(zip(aligned_raw, gold_aligned)):
        if align_raw_char == gold_raw_char and align_raw_char in [",", ".", "!", "?", ":", ";"]:
            if idx+2 < len(aligned_raw) :
                if aligned_raw[idx+1] == " " and (aligned_raw[idx+2] in alphabet):
                    print(aligned_raw[start_idx:idx+1])
                    remove_list = []
                    raw_remove_list = []
                    gold_remove_list = []
                    for temp_idx, (char_raw, char_gold) in enumerate(zip(aligned_raw[start_idx:idx+1], gold_aligned[start_idx:idx+1])):
                        if char_raw == "#" or char_gold == "#":
                            remove_list.append(temp_idx)
                        elif char_raw == "@":
                            raw_remove_list.append(temp_idx)
                        elif char_gold == "@":
                            gold_remove_list.append(temp_idx)
                    list_raw = [char for idx, char in enumerate(aligned_raw[start_idx:idx+1]) if idx not in remove_list+raw_remove_list+gold_remove_list]
                    list_gold = [char for idx, char in enumerate(gold_aligned[start_idx:idx+1]) if idx not in remove_list+gold_remove_list]
                    raw_text = "".join(list_raw).strip()
                    gold_text = "".join(list_gold).strip()
                    raw_text = noise_functions.noise_pipe(raw_text, delete_prob=0.1,
                                                   replace_prob=0.1, insert_prob=0.1, swap_prob=0.1,
                                                   repeat_prob=0.1, insert_linefill_prob=0.1, permutation_range=3)
                    if textdistance.hamming.normalized_similarity(raw_text, gold_text) < 0.1:
                        processed_raw.append(raw_text)
                        processed_gold.append(gold_text)
                    start_idx = idx+1
    return processed_raw, processed_gold

def load_ICDAR(base_addr, max_length=512):
    paired_data = {
        "train": {"text1":[], "text2":[], "label":[]},
        "dev": {"text1":[], "text2":[], "label":[]}
    }
    for set in ["train","dev"]:
        ocred = []
        correction = []
        basedir = base_addr + f"/{set}"
        for dir_name in ["EN1", "eng_monograph", "eng_periodical"]:
            tempdir = basedir + f"/{dir_name}"
            filelist = os.listdir(tempdir)
            for file in filelist:
                if file.endswith(".txt"):
                    with open(tempdir + "/" + file, "r") as f:
                        text = f.read()
                        raw, aligned_raw, aligned_gold = text.split("\n")[0], text.split("\n")[1], text.split("\n")[2]
                        processed_raw, processed_gold = remove_alignment_errors(raw, aligned_raw, aligned_gold)
                    ocred.extend(processed_raw)
                    correction.extend(processed_gold)
        paired_data[set]["text1"] = ocred
        paired_data[set]["text2"] = correction
        paired_data[set]["label"] = [1 for i in range(len(ocred))]
    nega_paired_data = icdar_negative_sampling(paired_data)
    dataset = datasets.DatasetDict({"train":datasets.Dataset.from_dict(paired_data["train"], split="train"),
                          "dev":datasets.Dataset.from_dict(paired_data["dev"], split="dev")})
    merged_train = datasets.concatenate_datasets([dataset["train"], nega_paired_data["train"]])
    merged_dev = datasets.concatenate_datasets([dataset["dev"], nega_paired_data["dev"]])
    merged_dataset = datasets.DatasetDict({"train":merged_train, "dev":merged_dev})
    return merged_dataset

def icdar_negative_sampling(paired_data):
    nega_paired_data = {
        "train": {"text1":[], "text2":[], "label":[]},
        "dev": {"text1":[], "text2":[], "label":[]}
    }
    for set in ["train", "dev"]:
        nega_sent1 = []
        nega_sent2 = []
        for idx, (sent1, sent2, label) in tqdm(enumerate(zip(paired_data[set]["text1"], paired_data[set]["text2"], paired_data[set]["label"]))):
            nega_sent1.extend(random.sample([paired_data[set]["text1"][i] for i in range(len(paired_data[set]["text1"])) if i != idx], 1))
            nega_sent2.append(sent2)
        nega_paired_data[set]["text1"] = nega_sent1
        nega_paired_data[set]["text2"] = nega_sent2
        nega_paired_data[set]["label"] = [0 for i in range(len(nega_sent1))]
    nega_data = datasets.DatasetDict({"train": datasets.Dataset.from_dict(nega_paired_data["train"], split="train"),
                          "dev": datasets.Dataset.from_dict(nega_paired_data["dev"], split="dev")})
    return nega_data

def noise_language_modelling(dataset_name="wikitext", dataset_type="wikitext-2-v1"):
    paired_data = {
        "train": {"text1":[], "text2":[], "label":[]},
        "dev": {"text1":[], "text2":[], "label":[]}
    }
    data = datasets.load_dataset(dataset_name, dataset_type)
    data = data.filter(lambda x: len(x["text"]) > 0)
    for set in ["train", "test"]:
        for text1 in data[set]["text"]:
            if set == "test":
                save_part = "dev"
            else:
                save_part = set
            paired_data[save_part]["text1"].append(text1)
            paired_data[save_part]["text2"].append(noise_functions.noise_pipe(text1, delete_prob=0.1,
                                                   replace_prob=0.1, insert_prob=0.1, swap_prob=0.1,
                                                   repeat_prob=0.1, insert_linefill_prob=0.1, permutation_range=3))
            paired_data[save_part]["label"].append(1)
    nega_paired_data = icdar_negative_sampling(paired_data)
    dataset = datasets.DatasetDict({"train":datasets.Dataset.from_dict(paired_data["train"], split="train"),
                            "dev":datasets.Dataset.from_dict(paired_data["dev"], split="dev")})
    merged_train = datasets.concatenate_datasets([dataset["train"], nega_paired_data["train"]])
    merged_dev = datasets.concatenate_datasets([dataset["dev"], nega_paired_data["dev"]])
    merged_dataset = datasets.DatasetDict({"train":merged_train, "dev":merged_dev})
    return merged_dataset

def noise_pretrain():
    wiki_text = noise_language_modelling()
    engpocr_text = load_ICDAR("EngPostOCR/")
    merged_train = datasets.concatenate_datasets([wiki_text["train"], engpocr_text["train"]], split="train")
    merged_dev = datasets.concatenate_datasets([wiki_text["dev"], engpocr_text["dev"]], split="dev")
    merged_dataset = datasets.DatasetDict({"train":merged_train, "dev":merged_dev})
    return merged_dataset

def train_all():
    news_copy = load_news_copy("NEWS-COPY/")
    noise_pretrain_dataset = noise_pretrain()

    merged_train = datasets.concatenate_datasets([news_copy["train"], noise_pretrain_dataset["train"]], split="train")
    merged_dev = datasets.concatenate_datasets([news_copy["dev"], noise_pretrain_dataset["dev"]], split="dev")
    merged_dataset = datasets.DatasetDict({"train":merged_train, "dev":merged_dev})
    return merged_dataset


def load_scholar():
    f = open("scholar/data.pkl", "rb")
    data = pickle.load(f)
    f.close()
    # data_set = []
    # for cluster in data:
    #     for pair in cluster:
    #         data_set.append({"text1":f"{pair[0]}\n{pair[1]}", "text2":f"{pair[4]}\n{pair[5]}", "label":1})
    #data = datasets.Dataset.from_list(data_set, split="train")
    return data


