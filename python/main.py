import argparse, csv, os, yaml
import pandas as pd

# NOTE/TODO: Do different imports based on specified language? (en or ru)
from utils.combine_en_data import (
    preprocess_conll,
    create_combined_en_dataset,
    map_to_standardized_labels,
    standardize_labels_and_save
)

if __name__ == '__main__':

    # Take in command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-lang', '--lang', dest = 'language', type = str,
                        default = 'en',
                        help = 'en to run the English model, ru for Russian')
    parser.add_argument('-cfg', '--config', dest = 'config_dir', type = str,
                        default = 'config/config.yml',
                        help = 'where the config file is located')
    args = parser.parse_args()

    # Set up configuration (see config/config.yml)
    config_path = os.path.join('..', args.config_dir)

    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    ###
    # Does the actual combining of EN data
    ###
    conll_path = '../' + cfg['conll_path']
    ee_path = '../' + cfg['ee_path']
    combined_path = '../' + cfg['en_combined_path']

    dataset_filenames = ['train_combined.txt', 'dev_combined.txt', 'test_combined.txt']
    dataset_file_list = [combined_path + fn for fn in dataset_filenames]

    # Training set
    create_combined_en_dataset([conll_path + 'train.txt',
                                ee_path + 'wnut17train.conll'],
                                combined_path + 'train_combined.txt')

    # Validation set
    create_combined_en_dataset([conll_path + 'valid.txt',
                                ee_path + 'emerging.dev.conll'],
                                combined_path + 'dev_combined.txt')

    # Test set
    create_combined_en_dataset([conll_path + 'test.txt',
                                ee_path + 'emerging.test.annotated'],
                                combined_path + 'test_combined.txt')

    # Standardize the labels on all 3 combined datasets
    standardize_labels_and_save(dataset_file_list)

    # TODO: Pandas FutureWarning about read_table deprecation
    # See if this can be relatively easily swapped out with read_csv

    ### Below this line will be the code to train or run the NER model
    ### Also, might add an arg into the ArgParser about plotting/exporting
    ### some stats about the dataset and labels
