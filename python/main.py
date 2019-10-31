import argparse
import csv
import datetime as dt
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import pandas as pd
from pytorch_transformers import BertTokenizer, BertForTokenClassification, BertConfig
import re
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix
import spacy
from spacy.gold import Doc, biluo_tags_from_offsets
import subprocess
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import yaml
from utils.main_utils import (
    get_args,
    process_en_data,
    process_ru_data,
    get_special_tokens,
    load_and_prepare_data,
    SentenceGetter,
    BertDataset,
    tokenize_and_preserve_labels,
    get_hyperparameters,
    flat_accuracy,
    annot_confusion_matrix,
    train_and_save_model,
)

if __name__ == "__main__":

    # Use ArgumentParser to take CL args
    args = get_args()

    # Set up configuration (see config/config.yml)
    with open(args.config_dir, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # Constants
    label_types = cfg[args.language]["label_types"]
    MAX_LEN = args.max_len
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    MODEL = cfg[args.language]["model"]
    THIS_RUN = dt.datetime.now().strftime("%m.%d.%Y, %H.%M.%S")
    MAX_GRAD_NORM = 1.0
    NUM_LABELS = len(label_types)
    FULL_FINETUNING = True

    # Process and combine data
    if args.preprocess and args.language == "en":
        process_en_data(cfg)
    elif args.preprocess and args.language == "ru":
        process_ru_data(cfg)

    # Create directory for storing our model checkpoints
    if not os.path.exists("../models"):
        os.mkdir("../models")

    os.mkdir(f"../models/{THIS_RUN}")

    # Specify device data for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL, do_lower_case=False)

    # Create dicts for mapping from labels to IDs and back
    tag2idx = {t: i for i, t in enumerate(label_types)}
    idx2tag = {i: t for t, i in tag2idx.items()}

    # Load and prepare data
    train_dataloader, valid_dataloader = load_and_prepare_data(
        cfg, args.language, tokenizer, MAX_LEN, BATCH_SIZE, tag2idx
    )
    print("Loaded training and validation data into DataLoaders.")

    # Initialize model
    model = BertForTokenClassification.from_pretrained(MODEL, num_labels=NUM_LABELS)
    model.to(device)
    print(f"Initialized model and moved it to {device}.")

    # Set hyperparameters (optimizer, weight decay, learning rate)
    optimizer_grouped_parameters = get_hyperparameters(model, FULL_FINETUNING)
    optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)
    print("Initialized optimizer and set hyperparameters.")

    # Fine-tune model and save checkpoint every epoch
    train_and_save_model(
        model,
        tokenizer,
        optimizer,
        args,
        idx2tag,
        tag2idx,
        THIS_RUN,
        MAX_GRAD_NORM,
        device,
        train_dataloader,
        valid_dataloader,
    )
