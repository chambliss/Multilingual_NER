# -*- coding: UTF-8 -*-
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_transformers import BertTokenizer, BertForTokenClassification, BertConfig
import streamlit as st
import torch
import yaml

# Run using `streamlit run demo.py`

# NOTE: Change CHK_PATH to your preferred model checkpoint path before running.
# NOTE 2: Argparse currently breaks the demo if you try to pass args in the command
# line. If you'd like to use the demo in Russian, change argparse_default_lang to 'ru'
# below.
#
# Example path:
# CHK_PATH = "../models/10.03.2019, 17.12.59/train_checkpoint_epoch_5.tar"
CHK_PATH = "YOUR_CHECKPOINT_PATH_HERE"
ARGPARSE_DEFAULT_LANG = "en"


@st.cache()
def load_model_and_tokenizer(chk_path, state_dict_name="model_state_dict", lang="en"):

    """
    Loads model from a specified checkpoint path. Replace `CHK_PATH` at the top of
    the script with where you are keeping the saved checkpoint.
    (Checkpoint must include a model state_dict, which by default is specified as
    'model_state_dict,' as it is in the `main.py` script.)
    """

    model_name = "bert-base-cased" if lang == "en" else "bert-base-multilingual-cased"

    checkpoint = torch.load(chk_path)
    model_state_dict = checkpoint[state_dict_name]
    model = BertForTokenClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS
    )
    model.load_state_dict(model_state_dict)
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-lang",
        "--lang",
        dest="language",
        type=str,
        default=ARGPARSE_DEFAULT_LANG,
        choices=["en", "ru"],
        help="en to load an English model, ru for Russian",
    )
    parser.add_argument(
        "-cfg",
        "--config",
        dest="config_dir",
        type=str,
        default="../config/config.yml",
        help="where the config file is located",
    )
    args = parser.parse_args()

    # Set up configuration (see config/config.yml)
    with open(args.config_dir, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    label_types = cfg[args.language]["label_types"]
    NUM_LABELS = len(label_types)

    # Create dicts for mapping from labels to IDs and back
    tag2idx = {t: i for i, t in enumerate(label_types)}
    idx2tag = {i: t for t, i in tag2idx.items()}

    ##
    # App functionality code begins here
    st.title("Predict Named Entities with BERT!")

    # Load in model and tokenizer
    model_load_state = st.text("Loading model...")
    model, tokenizer = load_model_and_tokenizer(CHK_PATH, lang=args.language)
    model_load_state.text("Loading model...done! (using st.cache)")

    # Set up demo text depending on language
    if args.language == "en":
        user_prompt = "What text do you want to predict on?"
        found_msg = "BERT found these named entities:"
        not_found_msg = (
            "BERT didn't find any entities. "
            + "\N{DISAPPOINTED FACE} Try a longer string."
        )
        plot_title = "Number of words/word-pieces matching each label:"
    else:
        user_prompt = "Какой текст вы хотите предсказать?"
        found_msg = "Модель нашла эти сущности:"
        not_found_msg = "Объекты не найдены. Попробуйте длинное предложение."
        plot_title = "Количество слов / частей слова, соответствующих каждому ярлыку:"

    default_input = cfg["demo_text"][args.language]
    user_input = st.text_area(user_prompt, value=default_input)

    try:
        model_output = predict_on_text(user_input)
        ents = [pair for pair in model_output if pair[1] != "O"]
    except IndexError:
        ents = None
        st.write(not_found_msg)

    if ents:
        # If entities found, generate a dataframe from the word pieces + labels
        ents_df = pd.DataFrame(ents)
        ents_df.columns = ["word_piece", "label"]
        ents_df["word_piece"] = ents_df["word_piece"].str.replace(" ", "")

        # Show the dataframe in table format
        st.write(found_msg)
        st.table(ents_df)

        # Create simple bar chart of label counts
        st.write(plot_title)
        st.bar_chart(ents_df["label"].value_counts().to_frame("count"))
