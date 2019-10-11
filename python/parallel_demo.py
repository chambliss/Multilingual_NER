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

    en_mgr = LanguageResourceManager("en", cfg, EN_CHK_PATH)
    ru_mgr = LanguageResourceManager("ru", cfg, RU_CHK_PATH)

    ru_input = create_input_prompt(ru_mgr)
    en_input = create_input_prompt(en_mgr)

    ru_bert_preds = ru_mgr.get_preds(ru_input, "bert")
    ru_spacy_preds = ru_mgr.get_preds(ru_input, "spacy")
    ru_viz_df = get_viz_df(ru_bert_preds, ru_spacy_preds)

    en_bert_preds = en_mgr.get_preds(en_input, "bert")
    en_spacy_preds = en_mgr.get_preds(en_input, "spacy")
    en_viz_df = get_viz_df(en_bert_preds, en_spacy_preds)

    explainer = """<pred style="background-color:#bdc9e1">Light blue</pred>: Entity
    predicted by one model<br><pred style="background-color:#74a9cf">Medium blue
    </pred>: Entity predicted by both models""".replace(
        "\n", ""
    )

    div_explainer = Div(text=explainer)
    div = produce_text_display(en_viz_df)
    div2 = produce_text_display(ru_viz_df)

    st.bokeh_chart(div_explainer)
    st.bokeh_chart(div)
    st.bokeh_chart(div2)
