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


@st.cache(allow_output_mutation=True)
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

    # If resulting string length is correct, create prediction columns for each tag
    # Future: Russian BERT also supports EVT and PRO labels - may
    # add support to display those within the demo later
    if len(str_rep) == len(toplevel_preds):
        preds_final = list(zip(str_rep, toplevel_preds))
        b_preds_df = pd.DataFrame(preds_final)
        b_preds_df.columns = ["text", "pred"]
        for tag in ["PER", "LOC", "ORG", "MISC"]:
            b_preds_df[f"b_pred_{tag.lower()}"] = np.where(
                b_preds_df["pred"].str.contains(tag), 1, 0
            )
        return b_preds_df.loc[:, "text":]
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

    # Match up tags between BERT and spaCy models (BERT left, spaCy right)
    # Note the English and Russian spaCy models support different tags.
    tag_matcher = {
        "PER": {"en": "PERSON", "ru": "PER"},
        "LOC": {},
        "ORG": {},
        "MISC": {"en": ["PRODUCT", "WORK_OF_ART"], "ru": "MISC"},
    }

    for tag in tag_matcher:
        if tag in ["PER", "MISC"]:
            did_predict_tag = s_preds_df["pred"].isin(tag_matcher[tag].values())
        elif tag in ["LOC", "ORG"]:
            did_predict_tag = s_preds_df["pred"] == tag

        s_preds_df[f"s_pred_{tag.lower()}"] = np.where(did_predict_tag, 1, 0)

    return s_preds_df.loc[:, "s_pred_per":]


# fmt: off
def create_pred_consistency_columns(combined_df):

    """
    Create columns w/ the model name if one model predicts an
    entity and the other doesn't; otherwise a blank string.
    These columns are used for highlighting the predictions via CSS.

    Condition 1: spaCy predicts no, BERT predicts yes
    Condition 2: spaCy predicts yes, BERT predicts no
    Condition 3: Both models agree

    (The "fmt" and "noqa" comments tell Black and flake8 to ignore minor
    stylistic violations here, in favor of increased readability.)
    """

    consistency_cols = []

    for tag in ["per", "loc", "org", "misc"]:
        cond1 = (combined_df[f"s_pred_{tag}"] == 0) & (combined_df[f"b_pred_{tag}"] == 1)
        cond2 = (combined_df[f"s_pred_{tag}"] == 1) & (combined_df[f"b_pred_{tag}"] == 0)
        cond3 = (combined_df[f"b_pred_{tag}"] == 1) & (combined_df[f"s_pred_{tag}"] == 1)

        which_model = np.where( # noqa
            cond1, "BERT",
                np.where(cond2, "spaCy",
                    np.where(cond3, "", ""))
        )

        consistency_col = pd.Series(which_model, name=f"model_name_{tag}")
        consistency_cols.append(consistency_col)

    return pd.concat(consistency_cols, axis=1)

# fmt: on
def get_viz_df(bert_preds_df, spacy_preds_df):

    """
    Joins the prediction dfs and produces the other columns needed for
    comparing/visualizing the models' predictions.
    """

    combined = pd.merge(
        bert_preds_df, spacy_preds_df, left_index=True, right_index=True
    )
    consistency_cols = create_pred_consistency_columns(combined)
    combined = pd.concat([combined, consistency_cols], axis=1)

    for tag in ["per", "loc", "org", "misc"]:
        pred_cols = [f"b_pred_{tag}", f"s_pred_{tag}"]
        combined[f"pred_sum_{tag}"] = combined[pred_cols].sum(axis=1)

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


def create_explainer(color_dict, ent_dict):

    explainer = """<b>Left:</b> Both BERT and spaCy found this entity <br>
        <b>Right:</b> Only one of BERT or spaCy found this entity <br><br>"""

    for ent_type in ent_dict:
        dark, light = color_dict[ent_dict[ent_type]]
        ent_html = f"""<b><span style="color: {dark}">{ent_type}</span>
        <span style="color: {light}">{ent_type}</span></b><br>"""
        explainer += ent_html

    return Div(text=explainer, width=500)


def produce_text_display(combined_pred_df, color_dict):

    """
    Returns a bokeh Div object containing the prediction DF's text as formatted
    HTML. The color of the word corresponds to the entity type, as defined in
    `color_dict` (which right now is pulled from `demo_colors` in config.yaml)

    Right now, `tooltip` is set to False by default because it is not supported
    in Streamlit.
    """

    def style_wrapper(s, tag, pred_score, model_name, tooltip=False):
        # Wraps a word that at least one model predicted to be an entity.
        dark, light = color_dict[tag]
        color = dark if pred_score == 2 else light
        model_name = "" if type(model_name) == float else model_name

        # note to self: using the "dark" color for the bg-color of the text is
        # generally too dark. Change this once hovers/tooltips are supported in
        # Streamlit.
        if tooltip:
            long_tag_names = {  # Define longer tag names for tooltip clarity
                "per": "PERSON",
                "loc": "LOCATION",
                "org": "ORGANIZATION",
                "misc": "MISC",
            }
            html = f"""<span class="pred" style="background-color: {color}">
            <span class="tooltip">
            {s}
            <span class="tooltiptext" style="background-color: {color}">
                <b>{long_tag_names[tag]}</b>
                <br>{model_name}
            </span>
            </span>
            </span>"""
        else:  # Simply change the inline color of the predicted word
            html = f"""<span style="color: {color}; font-weight: bold">
            {s}</span>
            """

        return html.replace("\n", "")

    text = []
    ps_cols = [col for col in combined_pred_df.columns if "pred_sum_" in col]

    # Iterates over each piece of text in the viz df, checking whether the text
    # was predicted to be at least one entity type. Grabs the tag name off the
    # end of the column name, then passes the necessary args to `style_wrapper`
    # to wrap that particular word in the styling HTML.
    for i, row in combined_pred_df.iterrows():

        if row[ps_cols].sum() > 0:
            row_no_text = row[ps_cols]
            tag_col = row_no_text[row_no_text > 0].index[0]
            tag = tag_col[-4:].replace("_", "")
            wrapped_text = style_wrapper(
                row["text"], tag, row[f"pred_sum_{tag}"], row[f"model_name_{tag}"]
            )
            text.append(wrapped_text)
        else:
            text.append(row["text"])

    # Future: once tooltips are supported, add tooltip CSS here.
    html_string = (
        """<div style="font-size: 18px; border-color: black">"""
        + " ".join(text)
        + "</div>"
    )

    return Div(text=html_string, width=700)
