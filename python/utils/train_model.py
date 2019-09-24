### Imports

import numpy as np

# Torch and training-related imports
from pytorch_transformers import BertTokenizer, BertForTokenClassification, BertConfig
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from seqeval.metrics import classification_report, f1_score
from sklearn.metrics import confusion_matrix

# Progress bar
from tqdm import tqdm, trange

###

# Data paths and other constants
# Place these in the YAML later
train_data_path = "../../data/en/combined/train_combined_std.txt"
dev_data_path = "../../data/en/combined/dev_combined_std.txt"
test_data_path = "../../data/en/combined/test_combined_std.txt"

label_types = [
    "B-PER",
    "I-PER",
    "B-LOC",
    "I-LOC",
    "B-ORG",
    "I-ORG",
    "B-MISC",
    "I-MISC",
    "O",
    "PAD",
]
MAX_LEN = 75
BATCH_SIZE = 32
EPOCHS = 10
MAX_GRAD_NORM = 1.0
NUM_LABELS = len(label_types)
FULL_FINETUNING = True

# Create dicts for mapping from labels to IDs and back
tag2idx = {t: i for i, t in enumerate(label_types)}
idx2tag = {i: t for t, i in tag2idx.items()}

# Specify device data for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)

###

## Useful functions and classes

# Create class for reading in and separating sentences from their labels
class SentenceGetter(object):
    def __init__(self, data_path):

        """
        Constructs a list of lists for sentences and labels
        from the data_path passed to SentenceGetter.

        We can then access sentences using the .sents
        attribute, and labels using .labels.
        """

        with open(data_path) as f:
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
            except IndexError:
                continue

            self.sents.append(toks)
            self.labels.append(labs)

        print(f"Constructed SentenceGetter with {len(self.sents)} examples.")


class BertDataset:
    def __init__(self, sg):

        """
        Takes care of the tokenization and ID-conversion steps
        for prepping data for BERT.

        Takes a SentenceGetter (sg) initialized on the data you
        want to use as argument.
        """

        # Tokenize the text into subwords in a label-preserving way
        tokenized_texts = [
            tokenize_and_preserve_labels(sent, labs)
            for sent, labs in zip(sg.sents, sg.labels)
        ]

        self.toks = [text[0] for text in tokenized_texts]
        self.labs = [text[1] for text in tokenized_texts]

        # Convert tokens to IDs
        self.input_ids = pad_sequences(
            [tokenizer.convert_tokens_to_ids(txt) for txt in self.toks],
            maxlen=MAX_LEN,
            dtype="long",
            truncating="post",
            padding="post",
        )

        # Convert tags to IDs
        self.tags = pad_sequences(
            [[tag2idx.get(l) for l in lab] for lab in self.labs],
            maxlen=MAX_LEN,
            value=tag2idx["PAD"],
            padding="post",
            dtype="long",
            truncating="post",
        )

        # Place a mask (zero) over the padding tokens
        self.attn_masks = [[float(i > 0) for i in ii] for ii in self.input_ids]


def tokenize_and_preserve_labels(sentence, text_labels):

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


def flat_accuracy(preds, labels):

    """
    Define a flat accuracy metric to use while training the model.
    """

    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def annot_confusion_matrix(valid_tags, pred_tags):

    """
    Create an annotated confusion matrix by adding label
    annotations and formatting to sklearn's `confusion_matrix`.
    """

    # Convert tag IDs to readable labels, create header
    predicted_or_true_tags = sorted(list(set(valid_tags + pred_tags)))
    header = [idx2tag[n] for n in predicted_or_true_tags]

    # Calculate the actual confusion matrix
    matrix = confusion_matrix(valid_tags, pred_tags)

    # Final formatting touches for the string output
    mat_formatted = [header[i] + "\t" + str(row) for i, row in enumerate(matrix)]
    content = "\t" + " ".join(header) + "\n" + "\n".join(mat_formatted)

    return content


###

# Read training and dev data in using SentenceGetter
getter_train = SentenceGetter(train_data_path)
getter_dev = SentenceGetter(dev_data_path)

# Prepare and convert the data to appropriate formats
train, dev = BertDataset(getter_train), BertDataset(getter_dev)

# Input IDs (tokens), tags (label IDs), attention masks
tr_inputs = torch.tensor(train.input_ids)
val_inputs = torch.tensor(dev.input_ids)
tr_tags = torch.tensor(train.tags)
val_tags = torch.tensor(dev.tags)
tr_masks = torch.tensor(train.attn_masks)
val_masks = torch.tensor(dev.attn_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)

print("Loaded training and validation data into DataLoaders successfully.")

###

# Initialize model
model = BertForTokenClassification.from_pretrained(
    "bert-base-cased", num_labels=NUM_LABELS
)

model.to(device)

print(f"Initialized model and moved it to {device}.")

###

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

###

epoch = 0
for _ in trange(EPOCHS, desc="Epoch"):
    epoch = epoch + 1
    # Training loop
    print("Starting training loop.")
    model.train()
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []

    for step, batch in enumerate(train_dataloader):

        print("Starting a new training batch.")

        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # forward pass
        outputs = model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels,
        )
        loss, tr_logits = outputs[:2]

        # backward pass
        loss.backward()

        # Compute train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

        # Compute training accuracy
        tr_logits = tr_logits.detach().cpu().numpy()
        tr_label_ids = b_labels.to("cpu").numpy()
        tr_preds.extend([list(p) for p in np.argmax(tr_logits, axis=2)])
        tr_labels.append(tr_label_ids)

        tmp_tr_accuracy = flat_accuracy(tr_logits, tr_label_ids)
        tr_accuracy += tmp_tr_accuracy

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=MAX_GRAD_NORM
        )

        # update parameters
        optimizer.step()
        model.zero_grad()

    tr_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps

    # Print training loss and accuracy per epoch
    print(f"Train loss: {tr_loss}")
    print(f"Train accuracy: {tr_accuracy}")

    print("Starting validation loop.")

    # VALIDATION on validation set
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

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1

    # Evaluate f1 score, loss, and accuracy on devset
    pred_tags = [label_types[p_i] for p in predictions for p_i in p]
    valid_tags = [label_types[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    cl_report = classification_report(valid_tags, pred_tags)
    conf_mat = annot_confusion_matrix(valid_tags, pred_tags)
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps

    # Report metrics
    print(f"Validation loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")
    print(f"Classification Report: {cl_report}")
    print(f"Confusion Matrix:\n {conf_mat}")

    # Save model and optimizer state_dict following every epoch
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
        f"train_checkpoint_epoch_{epoch}.tar",
    )
