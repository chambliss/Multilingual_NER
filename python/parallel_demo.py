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
    produce_text_display,
    create_explainer,
)
import numpy as np
import os
import pandas as pd
from pytorch_transformers import BertTokenizer, BertForTokenClassification, BertConfig
import spacy
import streamlit as st
import torch
import yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    st.title("Predict Entities on Parallel Text with BERT!")

    available_chkpts = [
        f"../models/{dir}/" + os.listdir(f"../models/{dir}")[0]
        for dir in os.listdir("../models/")
        if dir != ".DS_Store"
    ]

    EN_CHK_PATH = st.selectbox("English model checkpoint:", tuple(available_chkpts))
    RU_CHK_PATH = st.selectbox("Russian model checkpoint:", tuple(available_chkpts))

    try:
        en_mgr = LanguageResourceManager("en", cfg, EN_CHK_PATH)
        ru_mgr = LanguageResourceManager("ru", cfg, RU_CHK_PATH)
    except RuntimeError:
        st.write("One of the selected checkpoints is not compatible.")
        st.write("Are you sure you have the right checkpoint for this language?")

    ru_input = create_input_prompt(ru_mgr)
    en_input = create_input_prompt(en_mgr)

    ru_bert_preds = ru_mgr.get_preds(ru_input, "bert")
    ru_spacy_preds = ru_mgr.get_preds(ru_input, "spacy")
    ru_viz_df = get_viz_df(ru_bert_preds, ru_spacy_preds)

    en_bert_preds = en_mgr.get_preds(en_input, "bert")
    en_spacy_preds = en_mgr.get_preds(en_input, "spacy")
    en_viz_df = get_viz_df(en_bert_preds, en_spacy_preds)

    st.subheader("Prediction Summary:")

    # Set up colors and HTML for the explainer and the predicted text
    color_dict = cfg["demo_colors"]
    ent_dict = {
        "Person": "per",
        "Location": "loc",
        "Organization": "org",
        "Misc": "misc",
    }
    en_display = produce_text_display(en_viz_df, color_dict)
    ru_display = produce_text_display(ru_viz_df, color_dict)
    explainer = create_explainer(color_dict, ent_dict)
    ent_types = list(ent_dict.keys())

    st.bokeh_chart(explainer)
    st.bokeh_chart(ru_display)
    st.bokeh_chart(en_display)

    st.subheader("Prediction Details Per Entity Type:")

    # Allow users to view detailed prediction breakdown for a chosen entity type
    selected_ent = st.selectbox("Entity type: ", [ent_type for ent_type in ent_dict])
    ent = ent_dict[selected_ent]

    # Display fine-grained model prediction columns for selected entity
    for df, lang in zip([ru_viz_df, en_viz_df], ["Russian", "English"]):
        st.write(f"Predictions from {lang} models:")
        mask = df[f"pred_sum_{ent}"].values > 0
        st.table(df[mask][["text", f"b_pred_{ent}", f"s_pred_{ent}"]])
