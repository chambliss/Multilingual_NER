# -*- coding: UTF-8 -*-
import argparse
from bokeh.models.widgets.markups import Div
import numpy as np
import pandas as pd
from pytorch_transformers import BertTokenizer, BertForTokenClassification, BertConfig
import spacy
import streamlit as st
import torch
import yaml


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
        model_name, num_labels=model_state_dict["classifier.weight"].shape[0]
    )
    model.load_state_dict(model_state_dict)
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)

    return model, tokenizer


@st.cache(ignore_hash=True)
def load_spacy(language):

    model = "en_core_web_lg" if language == "en" else "xx_ent_wiki_sm"

    return spacy.load(model)


def get_bert_pred_df(model, tokenizer, input_text, label_dict):

    """
    Uses the model to make a prediction, with batch size 1.
    """

    encoded_text = tokenizer.encode(input_text)
    wordpieces = [tokenizer.decode(tok).replace(" ", "") for tok in encoded_text]

    input_ids = torch.tensor(encoded_text).unsqueeze(0).long()
    labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0).long()
    outputs = model(input_ids, labels=labels)
    loss, scores = outputs[:2]
    scores = scores.detach().numpy()

    label_ids = np.argmax(scores, axis=2)
    preds = [label_dict[i] for i in label_ids[0]]

    wp_preds = list(zip(wordpieces, preds))
    toplevel_preds = [pair[1] for pair in wp_preds if "##" not in pair[0]]
    str_rep = " ".join([t[0] for t in wp_preds]).replace(" ##", "").split()

    if len(str_rep) == len(toplevel_preds):
        preds_final = list(zip(str_rep, toplevel_preds))
        b_preds_df = pd.DataFrame(preds_final)
        b_preds_df.columns = ["text", "pred"]
        b_preds_df["b_pred_per"] = np.where(
            b_preds_df["pred"].str.contains("PER"), 1, 0
        )
        return b_preds_df[["text", "b_pred_per"]]
    else:
        print("Could not match up output string with preds.")
        return None


def get_spacy_pred_df(spacy_model, text):

    doc = spacy_model(text)
    tokens = [tok for tok in doc if not tok.is_space]

    s_preds_df = pd.DataFrame([(tok.text, tok.ent_type_) for tok in tokens])

    # Iterate over the spacy DF to make sure it'll be the same length as the BERT DF
    # (Differences arise due to differing tokenization conventions)
    # Also, convert "unconventional apostrophes" to regular ones
    texts = []
    preds = []
    for tup in s_preds_df.itertuples():
        if tup[1] == "'s" or bytes(tup[1], "utf-8") == b"\xe2\x80\x99s":
            texts.append("'")
            texts.append("s")
            preds.extend([tup[2]] * 2)
        elif "." in tup[1] and len(tup[1]) > 1:
            texts.append(tup[1][:-1])
            texts.append(".")
            preds.extend([tup[2]] * 2)
        else:
            texts.append(tup[1])
            preds.append(tup[2])

    s_preds_df = pd.DataFrame([texts, preds]).T
    s_preds_df.columns = ["text", "pred"]
    s_preds_df["s_pred_per"] = np.where(s_preds_df["pred"] == "PERSON", 1, 0)

    return s_preds_df[["text", "s_pred_per"]]


def create_pred_consistency_column(combined_df):

    """
    Create a column w/ the model name if one model predicts a
    person and the other doesn't; otherwise a blank string.
    This column is used later in visualizing the predictions.
    """

    cond1 = (combined_df["s_pred_per"] == 0) & (combined_df["b_pred_per"] == 1)
    cond2 = (combined_df["s_pred_per"] == 1) & (combined_df["b_pred_per"] == 0)
    cond3 = (combined_df["b_pred_per"] == 1) & (combined_df["s_pred_per"] == 1)

    which_model = np.where(
        cond1, "BERT", np.where(cond2, "spaCy", np.where(cond3, "", ""))
    )

    return pd.Series(which_model, name="model_name")


def get_viz_df(bert_preds_df, spacy_preds_df):

    """
    Joins the prediction dfs and produces the other columns needed for
    comparing/visualizing the models' predictions.
    """

    combined = pd.merge(
        bert_preds_df, spacy_preds_df["s_pred_per"], left_index=True, right_index=True
    )
    consistency_col = create_pred_consistency_column(combined)
    combined = pd.concat([combined, consistency_col], axis=1)
    combined["pred_sum"] = combined[["b_pred_per", "s_pred_per"]].sum(axis=1)

    return combined


def create_input_prompt(mgr):
    info_text = "Russian text" if mgr.lang == "ru" else "English translation"
    return st.text_area(f"Add the {info_text} here:", value=mgr.default_text)


class LanguageResourceManager:

    """
    Manages resources for each language, such as the models. Also acts as a
    convenient interface for getting predictions.
    """

    def __init__(self, lang, config, chk_path):

        self.lang = lang
        self.label_types = config[lang]["label_types"]
        self.idx2tag = idx2tag = {i: t for i, t in enumerate(self.label_types)}
        self.num_labels = len(self.label_types)
        self.bert_model, self.bert_tokenizer = load_model_and_tokenizer(
            chk_path, lang=lang
        )
        self.spacy_model = load_spacy(lang)
        self.default_text = config["parallel_demo_text"][lang]

    def get_preds(self, input_text, model="bert"):

        if model == "bert":
            return get_bert_pred_df(
                self.bert_model, self.bert_tokenizer, input_text, self.idx2tag
            )
        else:
            return get_spacy_pred_df(self.spacy_model, input_text)


def produce_text_display(combined_pred_df):

    """
    Returns a bokeh Div object containing the prediction DF's text as formatted
    HTML. A word is highlighted in dark blue if it was predicted to be a person
    by both models, and in light blue if only one model predicted "person."
    """

    def style_wrapper(s, pred_score):

        colors = {1: "#bdc9e1", 2: "#74a9cf"}
        color = colors[pred_score]

        return f"""<pred style="background-color:{color}">{s}</pred>"""

    text = []
    for tup in combined_pred_df.itertuples():
        if tup[5] > 0:
            text.append(style_wrapper(tup[1], tup[5]))
        else:
            text.append(tup[1])

    html_string = " ".join(text)

    return Div(text=html_string, width=700)
