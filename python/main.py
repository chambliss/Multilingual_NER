import argparse
import csv
import datetime as dt
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import pandas as pd
from pytorch_transformers import BertTokenizer, BertForTokenClassification, BertConfig
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix
from spacy import load
from spacy.gold import Doc, biluo_tags_from_offsets
import subprocess
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import yaml

# NOTE/TODO: Do different imports based on specified language? (en or ru)
from utils.combine_en_data import (
    preprocess_conll,
    create_combined_en_dataset,
    map_to_standardized_labels,
    standardize_labels_and_save,
)

from utils.combine_ru_data import open_file, process_pair, process_and_save_data

from utils.train import (
    SentenceGetter,
    BertDataset,
    tokenize_and_preserve_labels,
    flat_accuracy,
    annot_confusion_matrix,
)

if __name__ == "__main__":

    # Take in command-line arguments (dest is the name used to reference parsed args
    # within the code)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-lang",
        "--lang",
        dest="language",
        type=str,
        default="en",
        choices=["en", "ru"],
        help="en to run the English model, ru for Russian",
    )
    parser.add_argument(
        "-cfg",
        "--config",
        dest="config_dir",
        type=str,
        default="../config/config.yml",
        help="where the config file is located",
    )
    parser.add_argument(
        "-max_len",
        "--max_len",
        dest="max_len",
        type=int,
        default=75,
        help="maximum length in WordPieces per training example sequence",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=32,
        help="training batch size",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        dest="epochs",
        type=int,
        default=5,
        help="number of epochs to train",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        type=bool,
        default=True,
        help="whether to print out the confusion matrix during training; "
        + "not recommended if language = 'ru'",
    )

    args = parser.parse_args()

    # Set up configuration (see config/config.yml)
    with open(args.config_dir, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    ###
    # Processes and combines data (either en or ru)
    ###
    if args.language == "en":
        ## TODO: This section violates DRY - clean up and shorten
        conll_path = cfg["conll_path"]
        ee_path = cfg["ee_path"]
        combined_path = cfg["en_combined_path"]

        dataset_filenames = [
            "train_combined.txt",
            "dev_combined.txt",
            "test_combined.txt",
        ]
        dataset_file_list = [combined_path + fn for fn in dataset_filenames]

        # Training set
        create_combined_en_dataset(
            [conll_path + "train.txt", ee_path + "wnut17train.conll"],
            combined_path + "train_combined.txt",
        )

        # Validation set
        create_combined_en_dataset(
            [conll_path + "valid.txt", ee_path + "emerging.dev.conll"],
            combined_path + "dev_combined.txt",
        )

        # Test set
        create_combined_en_dataset(
            [conll_path + "test.txt", ee_path + "emerging.test.annotated"],
            combined_path + "test_combined.txt",
        )

        # Standardize the labels on all 3 combined datasets
        standardize_labels_and_save(dataset_file_list)

    # Valid bc 'en' and 'ru' are the only allowed choices for arg parser
    else:
        process_and_save_data(cfg["ru_data_path"])

    ### Training code starts here

    label_types = cfg[args.language]["label_types"]
    MAX_LEN = args.max_len
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    MODEL = cfg[args.language]["model"]
    THIS_RUN = dt.datetime.now().strftime("%m.%d.%Y, %H.%M.%S")
    MAX_GRAD_NORM = 1.0
    NUM_LABELS = len(label_types)
    FULL_FINETUNING = True

    # Create directory for storing our model checkpoints
    subprocess.run(["mkdir", f"../models/{THIS_RUN}"])

    # Specify device data for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL, do_lower_case=False)

    # Create dicts for mapping from labels to IDs and back
    tag2idx = {t: i for i, t in enumerate(label_types)}
    idx2tag = {i: t for t, i in tag2idx.items()}

    # For use in preprocessing (see BertDataset class) and filtering
    # only for non-padding predictions during calculation of metrics
    pad_tok = tokenizer.vocab["[PAD]"]
    sep_tok = tokenizer.vocab["[SEP]"]
    cls_tok = tokenizer.vocab["[CLS]"]
    o_lab = tag2idx["O"]

    # Load and prepare data
    train_data_path = cfg[args.language]["train_data_path"]
    dev_data_path = cfg[args.language]["dev_data_path"]

    getter_train = SentenceGetter(train_data_path, tag2idx)
    getter_dev = SentenceGetter(dev_data_path, tag2idx)
    train = BertDataset(getter_train, tokenizer, MAX_LEN, tag2idx)
    dev = BertDataset(getter_dev, tokenizer, MAX_LEN, tag2idx)

    # Input IDs (tokens), tags (label IDs), attention masks
    tr_inputs = torch.tensor(train.input_ids)
    val_inputs = torch.tensor(dev.input_ids)
    tr_tags = torch.tensor(train.tags)
    val_tags = torch.tensor(dev.tags)
    tr_masks = torch.tensor(train.attn_masks)
    val_masks = torch.tensor(dev.attn_masks)

    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=BATCH_SIZE
    )

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(
        valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE
    )

    print("Loaded training and validation data into DataLoaders.")

    # Initialize model
    model = BertForTokenClassification.from_pretrained(MODEL, num_labels=NUM_LABELS)
    model.to(device)

    print(f"Initialized model and moved it to {device}.")

    # Set hyperparameters (optimizer, weight decay, learning rate)

    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

    print("Initialized optimizer and set hyperparameters.")

### Training and evaluation begins here

epoch = 0
for _ in trange(EPOCHS, desc="Epoch"):
    epoch += 1

    # Training loop
    print("Starting training loop.")
    model.train()
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []

    for step, batch in enumerate(train_dataloader):

        # Add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Forward pass
        outputs = model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels,
        )
        loss, tr_logits = outputs[:2]

        # Backward pass
        loss.backward()

        # Compute train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

        # Subset out unwanted predictions on CLS/PAD/SEP tokens
        preds_mask = (
            (b_input_ids != cls_tok)
            & (b_input_ids != pad_tok)
            & (b_input_ids != sep_tok)
        )

        tr_logits = tr_logits.detach().cpu().numpy()
        tr_label_ids = torch.masked_select(b_labels, (preds_mask == 1))
        tr_batch_preds = np.argmax(tr_logits[preds_mask.squeeze()], axis=1)
        tr_batch_labels = tr_label_ids.to("cpu").numpy()
        tr_preds.extend(tr_batch_preds)
        tr_labels.extend(tr_batch_labels)

        # Compute training accuracy
        tmp_tr_accuracy = flat_accuracy(tr_batch_labels, tr_batch_preds)
        tr_accuracy += tmp_tr_accuracy

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=MAX_GRAD_NORM
        )

        # Update parameters
        optimizer.step()
        model.zero_grad()

    tr_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps

    # Print training loss and accuracy per epoch
    print(f"Train loss: {tr_loss}")
    print(f"Train accuracy: {tr_accuracy}")

    # Validation loop
    print("Starting validation loop.")

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions, true_labels = [], []

    for batch in valid_dataloader:

        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )
            tmp_eval_loss, logits = outputs[:2]

        # Subset out unwanted predictions on CLS/PAD/SEP tokens
        preds_mask = (
            (b_input_ids != cls_tok)
            & (b_input_ids != pad_tok)
            & (b_input_ids != sep_tok)
        )

        logits = logits.detach().cpu().numpy()
        label_ids = torch.masked_select(b_labels, (preds_mask == 1))
        val_batch_preds = np.argmax(logits[preds_mask.squeeze()], axis=1)
        val_batch_labels = label_ids.to("cpu").numpy()
        predictions.extend(val_batch_preds)
        true_labels.extend(val_batch_labels)

        tmp_eval_accuracy = flat_accuracy(val_batch_labels, val_batch_preds)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1

    # Evaluate loss, acc, conf. matrix, and class. report on devset
    pred_tags = [idx2tag[i] for i in predictions]
    valid_tags = [idx2tag[i] for i in true_labels]
    cl_report = classification_report(valid_tags, pred_tags)
    conf_mat = annot_confusion_matrix(valid_tags, pred_tags)
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps

    # Report metrics
    print(f"Validation loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")
    print(f"Classification Report:\n {cl_report}")
    if args.verbose:
        print(f"Confusion Matrix:\n {conf_mat}")

    # Save model and optimizer state_dict following every epoch
    # TODO: Make this more robust by creating a timestamped directory in 'models'
    # for each new run of the script - could save logs here too if/when logging
    # is implemented
    save_path = f"../models/{THIS_RUN}/train_checkpoint_epoch_{epoch}.tar"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": tr_loss,
            "train_acc": tr_accuracy,
            "eval_loss": eval_loss,
            "eval_acc": eval_accuracy,
            "classification_report": cl_report,
            "confusion_matrix": conf_mat,
        },
        save_path,
    )
    print(f"Checkpoint saved to {save_path}.")
