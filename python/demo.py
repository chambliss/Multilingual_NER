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
    create_pred_consistency_columns,
    get_viz_df,
    create_explainer,
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

    try:
        mgr = LanguageResourceManager(args.lang, cfg, CHK_PATH)
    except RuntimeError:
        st.write("The selected checkpoint is not compatible with this BERT model.")
        st.write("Are you sure you have the right checkpoint?")

    user_prompt = "What text do you want to predict on?"
    default_input = cfg["demo_text"][args.lang]
    user_input = st.text_area(user_prompt, value=default_input)

    # Produce and align predictions from both models
    bert_preds = mgr.get_preds(user_input, "bert")
    spacy_preds = mgr.get_preds(user_input, "spacy")
    viz_df = get_viz_df(bert_preds, spacy_preds)

    st.subheader("Prediction Summary:")

    # Set up colors and HTML for the explainer and the predicted text
    color_dict = cfg["demo_colors"]
    ent_dict = {
        "Person": "per",
        "Location": "loc",
        "Organization": "org",
        "Misc": "misc",
    }
    display = produce_text_display(viz_df, color_dict)
    explainer = create_explainer(color_dict, ent_dict)
    ent_types = list(ent_dict.keys())

    # Display the explainer and predicted text
    st.bokeh_chart(explainer)
    st.bokeh_chart(display)

    st.subheader("Prediction Details Per Entity Type:")

    # Allow users to view detailed prediction breakdown for a chosen entity type
    selected_ent = st.selectbox("Entity type: ", [ent_type for ent_type in ent_dict])
    ent = ent_dict[selected_ent]
    st.write(f"Prediction summary for {selected_ent}: ")

    # Display fine-grained model prediction columns for selected entity
    mask = viz_df[f"pred_sum_{ent}"].values > 0
    st.table(viz_df[mask][["text", f"b_pred_{ent}", f"s_pred_{ent}"]])
