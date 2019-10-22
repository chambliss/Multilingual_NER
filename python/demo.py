# -*- coding: UTF-8 -*-
import argparse
from bokeh.models.widgets.markups import Div
from utils.demo_utils import (
    LanguageResourceManager,
    load_model_and_tokenizer,
    load_spacy,
    create_input_prompt,
    get_bert_pred_df,
    get_spacy_pred_df,
    create_pred_consistency_column,
    get_viz_df,
    produce_text_display,
)
import numpy as np
import os
import pandas as pd
from pytorch_transformers import BertTokenizer, BertForTokenClassification, BertConfig
import spacy
import streamlit as st
import torch
import yaml

# Run using `streamlit run demo.py en`
# For the Russian demo, change en to ru
# Do NOT pass `lang` or `config` using - or --. Streamlit is currently
# not compatible with this argument format.

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "lang",
        type=str,
        default="en",
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

    ##
    # App functionality code begins here
    st.title("Predict Named Entities with BERT!")

    available_chkpts = [
        f"../models/{dir}/" + os.listdir(f"../models/{dir}")[0]
        for dir in os.listdir("../models/")
        if dir != ".DS_Store"
    ]

    # Create selectbox for users to select checkpoint
    CHK_PATH = st.selectbox("Model checkpoint:", tuple(available_chkpts))

    # Load in models and tokenizer
    try:
        model_load_state = st.text("Loading models...")
        model, tokenizer = load_model_and_tokenizer(CHK_PATH, lang=args.lang)
        nlp = load_spacy(args.lang)
        model_load_state.text("Loading models...done!")
    except RuntimeError:
        st.write("The selected checkpoint is not compatible with this BERT Model.")
        st.write("Are you sure you have the right checkpoint?")

    mgr = LanguageResourceManager(args.lang, cfg, CHK_PATH)

    user_prompt = "What text do you want to predict on?"
    default_input = cfg["demo_text"][args.lang]
    user_input = st.text_area(user_prompt, value=default_input)

    # Produce and align predictions from both models
    bert_preds = mgr.get_preds(user_input, "bert")
    spacy_preds = mgr.get_preds(user_input, "spacy")
    viz_df = get_viz_df(bert_preds, spacy_preds)
    div = produce_text_display(viz_df)

    # Show the highlighted HTML output of the input text
    explainer = """<pred style="background-color:#bdc9e1">Light blue</pred>: Entity
    predicted by one model<br><pred style="background-color:#74a9cf">Medium blue
    </pred>: Entity predicted by both models""".replace(
        "\n", ""
    )

    div_explainer = Div(text=explainer)
    st.bokeh_chart(div_explainer)
    st.bokeh_chart(div)

    entity_type = ["PER", "PERSON"]
    st.write("Prediction summary:")
    st.write(viz_df[viz_df["pred_sum"] > 0])

    msg = "(NOTE: this demo currently only supports Person entity predictions.)"
    st.write(msg)
