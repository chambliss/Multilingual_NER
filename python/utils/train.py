## Imports
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from pytorch_transformers import BertTokenizer, BertForTokenClassification, BertConfig
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix
import torch

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
