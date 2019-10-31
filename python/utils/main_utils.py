## Imports
import argparse
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from pytorch_transformers import BertTokenizer, BertForTokenClassification, BertConfig
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange


# Create class for reading in and separating sentences from their labels
class SentenceGetter(object):
    def __init__(self, data_path, tag2idx):

        """
        Constructs a list of lists for sentences and labels
        from the data_path passed to SentenceGetter.

        We can then access sentences using the .sents
        attribute, and labels using .labels.
        """

        with open(data_path) as f:
            if "ru" in data_path:
                txt = f.read().split("\n\n")
            else:
                txt = f.read().split("\n \n")

        self.sents_raw = [(sent.split("\n")) for sent in txt]
        self.sents = []
        self.labels = []

        for sent in self.sents_raw:
            tok_lab_pairs = [pair.split() for pair in sent]

            # Handles (very rare) formatting issue causing IndexErrors
            try:
                toks = [pair[0] for pair in tok_lab_pairs]
                labs = [pair[1] for pair in tok_lab_pairs]

                # In the Russian data, a few invalid labels such as '-' were produced
                # by the spaCy preprocessing. Because of that, we generate a mask to
                # check if there are any invalid labels in the sequence, and if there
                # are, we reindex `toks` and `labs` to exclude them.
                mask = [False if l not in tag2idx else True for l in labs]
                if any(mask):
                    toks = list(np.array(toks)[mask])
                    labs = list(np.array(labs)[mask])

            except IndexError:
                continue

            self.sents.append(toks)
            self.labels.append(labs)

        print(f"Constructed SentenceGetter with {len(self.sents)} examples.")


class BertDataset:
    def __init__(self, sg, tokenizer, max_len, tag2idx):

        """
        Takes care of the tokenization and ID-conversion steps
        for prepping data for BERT.

        Takes a SentenceGetter (sg) initialized on the data you
        want to use as argument.
        """

        pad_tok = tokenizer.vocab["[PAD]"]
        sep_tok = tokenizer.vocab["[SEP]"]
        o_lab = tag2idx["O"]

        # Tokenize the text into subwords in a label-preserving way
        tokenized_texts = [
            tokenize_and_preserve_labels(sent, labs, tokenizer)
            for sent, labs in zip(sg.sents, sg.labels)
        ]

        self.toks = [["[CLS]"] + text[0] for text in tokenized_texts]
        self.labs = [["O"] + text[1] for text in tokenized_texts]

        # Convert tokens to IDs
        self.input_ids = pad_sequences(
            [tokenizer.convert_tokens_to_ids(txt) for txt in self.toks],
            maxlen=max_len,
            dtype="long",
            truncating="post",
            padding="post",
        )

        # Convert tags to IDs
        self.tags = pad_sequences(
            [[tag2idx.get(l) for l in lab] for lab in self.labs],
            maxlen=max_len,
            value=tag2idx["O"],
            padding="post",
            dtype="long",
            truncating="post",
        )

        # Swaps out the final token-label pair for ([SEP], O)
        # for any sequences that reach the MAX_LEN
        for voc_ids, tag_ids in zip(self.input_ids, self.tags):
            if voc_ids[-1] == pad_tok:
                continue
            else:
                voc_ids[-1] = sep_tok
                tag_ids[-1] = o_lab

        # Place a mask (zero) over the padding tokens
        self.attn_masks = [[float(i > 0) for i in ii] for ii in self.input_ids]


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):

    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


def flat_accuracy(valid_tags, pred_tags):

    """
    Define a flat accuracy metric to use while training the model.
    """

    return (np.array(valid_tags) == np.array(pred_tags)).mean()


def annot_confusion_matrix(valid_tags, pred_tags):

    """
    Create an annotated confusion matrix by adding label
    annotations and formatting to sklearn's `confusion_matrix`.
    """

    # Create header from unique tags
    header = sorted(list(set(valid_tags + pred_tags)))

    # Calculate the actual confusion matrix
    matrix = confusion_matrix(valid_tags, pred_tags, labels=header)

    # Final formatting touches for the string output
    mat_formatted = [header[i] + "\t" + str(row) for i, row in enumerate(matrix)]
    content = "\t" + " ".join(header) + "\n" + "\n".join(mat_formatted)

    return content


def str2bool(v):

    """
    Helper function to make boolean args work correctly in `argparse.`
    """

    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():

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
        type=str2bool,
        default=True,
        help="whether to print out the confusion matrix during training",
    )
    parser.add_argument(
        "-pp",
        "--preprocess",
        dest="preprocess",
        type=str2bool,
        default=True,
        help="whether to preprocess language-specific data prior to training the model",
    )

    return parser.parse_args()


def process_en_data(cfg):

    """
    The 'main' processing function for preprocessing and combining the CONLL
    and EE data prior to model training.

    Takes the cfg dict from the yaml file in `main` so it knows where to look for the
    data directories.
    """

    # Move imports here to reduce clutter in main.py
    from utils.combine_en_data import (
        preprocess_conll,
        create_combined_en_dataset,
        map_to_standardized_labels,
        standardize_labels_and_save,
    )

    conll_path = cfg["conll_path"]
    ee_path = cfg["ee_path"]
    combined_path = cfg["en_combined_path"]

    dataset_filenames = ["train_combined.txt", "dev_combined.txt", "test_combined.txt"]
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


def process_ru_data(cfg):

    """
    The 'main' processing function for preprocessing and combining the FactRu
    and Shared Task 2019 data prior to model training.

    Takes the cfg dict from the yaml file in `main` so it knows where to look for the
    data directories.
    """

    # Move imports here to reduce clutter in main.py
    from utils.combine_ru_data import (
        FACTRU_LABEL_DICT,
        open_file,
        process_pair,
        process_pair_ST,
        process_and_save_factRu,
        prep_st_data,
        find_exact_matches,
        process_and_append_ST,
        cleanup,
    )

    process_and_save_factRu(
        cfg["factRu_path"], cfg["ru_combined_path"], FACTRU_LABEL_DICT
    )
    process_and_append_ST(cfg["st_path"], cfg["ru_combined_path"])
    cleanup(cfg["ru_combined_path"])


def get_special_tokens(tokenizer, tag2idx):

    pad_tok = tokenizer.vocab["[PAD]"]
    sep_tok = tokenizer.vocab["[SEP]"]
    cls_tok = tokenizer.vocab["[CLS]"]
    o_lab = tag2idx["O"]

    return pad_tok, sep_tok, cls_tok, o_lab


def load_and_prepare_data(cfg, lang, tokenizer, max_len, batch_size, tag2idx):

    train_data_path = cfg[lang]["train_data_path"]
    dev_data_path = cfg[lang]["dev_data_path"]

    getter_train = SentenceGetter(train_data_path, tag2idx)
    getter_dev = SentenceGetter(dev_data_path, tag2idx)
    train = BertDataset(getter_train, tokenizer, max_len, tag2idx)
    dev = BertDataset(getter_dev, tokenizer, max_len, tag2idx)

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
        train_data, sampler=train_sampler, batch_size=batch_size
    )

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(
        valid_data, sampler=valid_sampler, batch_size=batch_size
    )

    return train_dataloader, valid_dataloader


def get_hyperparameters(model, ff):

    # ff: full_finetuning
    if ff:
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

    return optimizer_grouped_parameters


def train_and_save_model(
    model,
    tokenizer,
    optimizer,
    parser_args,
    idx2tag,
    tag2idx,
    this_run,
    max_grad_norm,
    device,
    train_dataloader,
    valid_dataloader,
):

    pad_tok, sep_tok, cls_tok, o_lab = get_special_tokens(tokenizer, tag2idx)
    verbose = parser_args.verbose
    epochs = parser_args.epochs

    epoch = 0
    for _ in trange(epochs, desc="Epoch"):
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
                parameters=model.parameters(), max_norm=max_grad_norm
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
        if verbose:
            print(f"Confusion Matrix:\n {conf_mat}")

        # Save model and optimizer state_dict following every epoch
        save_path = f"../models/{this_run}/train_checkpoint_epoch_{epoch}.tar"
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
