import csv
import numpy as np
import pandas as pd

def preprocess_conll(data):

    """
    Quick preprocessing on the CONLL data so it can be combined
    with the Emerging Entities data (which doesn't need any
    preprocessing).

    Takes `data`, which is a list of strings read in from
    a CONLL data file.
    """

    # Remove DOCSTART lines to make CONLL data consistent
    # with the Emerging Entities dataset
    data = [line for line in data if 'DOCSTART' not in line]

    # Add appropriate tabbing and spacing to match EE data
    data = ['\t'.join([line.split()[0], line.split()[3]]) + '\n'
            if line != '\n'
            else line
            for line in data]

    return data

def create_combined_en_dataset(dataset_path_list, combined_path):

    """
    Takes a dataset_path_list of the two English datasets (can be edited
    to accommodate more datasets later), and a combined_path, which
    is a path string describing where to save the data.

    Combines the two English datasets such that they have the same formatting;
    specifically, each line should look like this: TOKEN\tLABEL\n.
    See example below.
     ['EU\tB-ORG\n',
     'rejects\tO\n',
     'German\tB-MISC\n',
     'call\tO\n',
     'to\tO\n',
     'boycott\tO\n',
     'British\tB-MISC\n',
     'lamb\tO\n',
     '.\tO\n',
     '\n', ...]
    """

    for path in dataset_path_list:
        # indicates that these are the CONLL files
        conll_paths = ['test.txt', 'train.txt', 'valid.txt']
        if path in ['../data/en/CONLL2003/' + p for p in conll_paths]:
            with open(path, 'r') as conll:
                conll_data = preprocess_conll(conll.readlines())

        else:
            with open(path, 'r') as ee:
                ee_data = ee.readlines()

    # Combine the two datasets
    ee_data.extend(conll_data)

    # Write out to specified path
    with open(combined_path, 'w+') as new:
        new.writelines(ee_data)

    # Print success message
    print('Combined {} and saved new dataset to {}.'
         .format(dataset_path_list, combined_path))

    return None


def map_to_standardized_labels(label):

    """
    Meant to be used w/ pd.apply().
    Maps a label to a standardized set of labels, because
    the CONLL and EE data include different labelsets and
    labeling conventions (EE has a larger # of classes,
    and writes out labels as "person", "location", etc.,
    while CONLL uses "PER", "LOC", and so on).
    """

    if pd.isna(label):
        return label

    # [:2] keeps the 'B-' or 'I-' part of the label
    elif 'loc' in label.lower():
        label = label[:2] + 'LOC'

    elif 'per' in label.lower():
        label = label[:2] + 'PER'

    elif any([s in label.lower() for s in ['org', 'corp', 'group']]):
        label = label[:2] + 'ORG'

    # For any leftover labels that are not 'O': map them to MISC
    elif label != 'O':
        label = label[:2] + 'MISC'

    return label


def standardize_labels_and_save(dataset_file_list):

    """
    Standardizes the labels for each dataset and saves them
    under the same filename + '_std' for 'standardized'.
    """

    for file in dataset_file_list:

        # `sep`, `quoting`, and skip_blank_lines args help preserve data structure
        data_df = pd.read_table(
                    file, header=None, skip_blank_lines=False,
                    sep=' |\t', quoting=csv.QUOTE_NONE, engine='python'
                    ).replace([None], np.nan)

        data_df[1] = data_df[1].apply(map_to_standardized_labels)

        data_df.to_csv(f'{file[:-4]}_std.txt', header=False, index=False,
                        sep=' ', quoting=csv.QUOTE_NONE)

        print(f'Saved standardized data to {file}.')

    return None
