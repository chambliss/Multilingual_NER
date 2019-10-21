# multilingual_NER

## About

This repository applies [BERT](https://github.com/google-research/bert) to named entity recognition in English and Russian in order to improve machine translation quality estimation between English-Russian sentence pairs. Named entities are a known challenge in machine translation, and this repository aims to quantify accuracy by comparing named entity recognition between Russian and English in translated sentence pairs.

## Installation

**NOTE**: Since there are quite a few requirements, creating a virtual environment for this repo is strongly recommended.

**Instructions**: Use `conda` to create the environment. Currently environment creation via pip is not supported.

To create the environment with `conda`, all you need to do is navigate to the `Multilingual_NER` directory and run the following in the command line:

`conda env create -f environment.yml`

You can verify that the required packages installed correctly by activating the environment (`conda activate m_ner`) and running `conda list`.

Partial list of dependencies:  

- pytorch-transformers 1.1.0+, torch 1.2.0+
- spaCy (2.1.0+)
- TensorFlow 1.14.0+, keras 2.2.4+ (only used for utility functions - will try to eliminate this dependency down the line)
- numpy 1.15.4+, pandas 0.25.0+
- pre-commit 1.18.0+
- streamlit 0.4.0+
- (Deployment dependencies - TBD)
- Miscellaneous minor packages: pathlib, seqeval, tqdm, yaml

## Usage

`python main.py` to train an English model and save per-epoch checkpoints.
Add `-lang ru` to train a Russian model.

Once you have a usable model checkpoint, open `python/demo.py` and set the path to the model checkpoint you'd like to use and the demo's default language. Then run `streamlit run demo.py en`, which opens a web app you can use to test the model's predictions interactively. For the Russian demo, change `en` to `ru`.

## Models

- The English model is a fine-tuned implementation of Google's `bert-base-cased` model, ensembled with spaCy's `en_core_web_lg`, which uses a CNN architecture.
- The Russian model is a fine-tuned implementation of Google's `bert-base-multilingual-cased` model, ensembled with spaCy's multilingual `xx_ent_wiki_sm` NER model, which uses a CNN architecture.
- Before running the demo or either of the spaCy evaluation notebooks, be sure to run `python -m spacy download MODEL_NAME`.

## Data Sources

- The English BERT model was fine-tuned on [CONLL2003](http://aclweb.org/anthology/W03-0419) data and [Emerging Entities '17](https://noisy-text.github.io/2017/emerging-rare-entities.html) data. The Emerging Entities '17 data is composed of informal text (such as tweets), and includes more diverse entity types (such as creative works and products) than CONLL2003, providing the model with the ability to identify MISC entities in addition to the standard person, location, and organization tags.
- The English spaCy model was trained on [OntoNotes](https://catalog.ldc.upenn.edu/LDC2013T19) and uses GloVe word vectors trained on [Common Crawl](https://commoncrawl.org/); see the spaCy docs for more information.
- The Russian model was fine-tuned on data from the [factRuEval-2016](https://github.com/dialogue-evaluation/factRuEval-2016/) shared task, as well as the [Balto-Slavic NLP 2019 Shared Task](http://bsnlp.cs.helsinki.fi/shared_task.html). Both tasks had NER as a focus, and factRuEval-2016 included a general fact-finding task.
- The multilingual spaCy model was trained on Nothman et al. (2010) Wikipedia corpus, and supports PER, LOC, ORG, and MISC entities. See [the spaCy docs](https://spacy.io/models/xx) for more info.
