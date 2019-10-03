# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_transformers import BertTokenizer, BertForTokenClassification, BertConfig
import streamlit as st
import torch

# NOTE: Change this to your preferred model checkpoint path before running
chk_path = "../models/checkpoints/train_checkpoint_epoch_2.tar"

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
]
NUM_LABELS = len(label_types)

# Create dicts for mapping from labels to IDs and back
tag2idx = {t: i for i, t in enumerate(label_types)}
idx2tag = {i: t for t, i in tag2idx.items()}


@st.cache()
def load_model_and_tokenizer(chk_path, state_dict_name="model_state_dict"):

    """
    Loads model from a specified checkpoint path. Replace `chk_path` at the top of
    the script with where you are keeping the saved checkpoint.
    (Checkpoint must include a model state_dict, which by default is specified as
    'model_state_dict,' as in the `main.py` script.)
    """

    checkpoint = torch.load(chk_path)
    model_state_dict = checkpoint[state_dict_name]
    model = BertForTokenClassification.from_pretrained(
        "bert-base-cased", num_labels=NUM_LABELS
    )
    model.load_state_dict(model_state_dict)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)

    return model, tokenizer


def predict_on_text(input_text):

    """
    Uses the model to make a prediction, with batch size 1.
    Returns a list of (word_piece, predicted_label) tuples.
    """

    encoded_text = tokenizer.encode(input_text)
    wordpieces = [tokenizer.decode(tok) for tok in encoded_text]
    input_ids = torch.tensor(encoded_text).unsqueeze(0).long()
    labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0).long()
    outputs = model(input_ids, labels=labels)
    loss, scores = outputs[:2]
    scores = scores.detach().numpy()
    label_ids = np.argmax(scores, axis=2)
    preds = [idx2tag[i] for i in label_ids[0]]

    return list(zip(wordpieces, preds))


# App functionality code begins here
st.title("Predict Named Entities with BERT!")

model_load_state = st.text("Loading model...")
model, tokenizer = load_model_and_tokenizer(chk_path)
model_load_state.text("Loading model...done! (using st.cache)")

default_input = """On this day in 1957, a California Superior Court judge ruled that
 "Howl", a poem by Allen Ginsberg, was of "redeeming social importance" and thus not
 obscene.""".replace(
    "\n", ""
)
user_input = st.text_area("What text do you want to predict on?", value=default_input)

try:
    model_output = predict_on_text(user_input)
    ents = [pair for pair in model_output if pair[1] != "O"]
except IndexError:
    ents = None
    "BERT didn't find any entities. " + "\N{DISAPPOINTED FACE} Try a longer string."

if ents:
    ents_df = pd.DataFrame(ents)
    ents_df.columns = ["word_piece", "label"]
    ents_df["word_piece"] = ents_df["word_piece"].str.replace(" ", "")

    st.write("BERT found these named entities:")
    st.table(ents_df)

    "How many of each label are in this example?"
    st.bar_chart(ents_df["label"].value_counts())
