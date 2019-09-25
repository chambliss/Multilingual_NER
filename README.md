# multilingual_NER

## About
Deployable [BERT](https://github.com/google-research/bert) ensemble models for named entity recognition in English and Russian. `multilingual_NER` provides a toolkit to train, validate, and deploy NER models with an interactive frontend and callable API.

## Installation
(will make available via `pip install` and `conda install` when project is closer to completion - hit the Watch button if you'd like to be updated on progress!)

## Usage
(also coming soon - check back in a few weeks!)

## Requirements
- **NOTE**: Since there are quite a few requirements, we recommend creating a virtual environment when using this package.
- pytorch-transformers (1.1.0+) (and torch)
- spaCy (2.1.0+)
- keras (and TensorFlow) (only used for utility functions - will try to eliminate this dependency down the line)
- numpy, pandas
- pre-commit
- (Any packages used for visualizing datasets and/or model performance in the frontend - will update once frontend is ready)
- (Probably Flask and any other deployment dependencies - will update once ready)
- Miscellaneous minor packages: pathlib, seqeval, tqdm, yaml

## Models
- The English model is a fine-tuned implementation of Google's `bert-base-cased` model, ensembled with spaCy's `en_core_web_lg`, which uses a CNN architecture.
- The Russian model is a fine-tuned implementation of Google's `bert-base-multilingual-cased` model, ensembled with spaCy's multilingual `xx_ent_wiki_sm` NER model, which uses a CNN architecture.

## Data Sources
The English model was fine-tuned on [CONLL2003](http://aclweb.org/anthology/W03-0419) data and [Emerging Entities '17](https://noisy-text.github.io/2017/emerging-rare-entities.html) data. The Emerging Entities '17 data is composed of informal text (such as tweets), and includes more diverse entity types (such as creative works and products) than CONLL2003, providing the model with the ability to identify MISC entities in addition to the standard person, location, and organization tags.

The Russian model was fine-tuned on data from the [factRuEval-2016](https://github.com/dialogue-evaluation/factRuEval-2016/) shared task, which focused on both NER and general fact-finding in Russian text.
