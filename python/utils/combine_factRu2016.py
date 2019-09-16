import csv
import numpy as np
import os
import pandas as pd
import re

def create_token_df(token_file_path):

    """
    Creates the DataFrame of tokens by reading the token file in as a DF.

    Ultimately, every token needs to be
    matched with a label (or a NaN representing "no label"), and structuring
    sequential tokens as a DF allows us to perform joins with span data (where
    the labels are defined) while retaining order.

    Takes a string defining a path to a *.tokens file from the factRuEval-2016
    dataset as argument.
    """

    token_col_names = ['id', 'begin_index', 'length', 'text']
    token_df = pd.read_csv(token_file_path, sep=' ', skip_blank_lines=False,
                           header=None, quoting=csv.QUOTE_NONE)
    token_df.columns = token_col_names

    return token_df


def extract_span_info(span_file_path):

    """
    Creats a list of lists contaning span data from a *.spans file.
    (We can't read it straight in as a DF because the line length
    is dependent on how many tokens are in the span.)

    (Some of the info is not kept in later steps, like token ids and
    phrase text, as they turned out not to be necessary for accurately
    joining the labels to the tokens. May be removed from the function
    later.)
    """

    with open(span_file_path) as s:
        info = s.readlines()

    intermediate = [l.split() for l in info]

    # Extract only the necessary fields for each intermediate representation
    # Grab token ids + phrase text, span id, mention type, position, and length of span/phrase
    fields = [(i_rep[7:], i_rep[0], i_rep[1], i_rep[2], i_rep[5])
              for i_rep in intermediate]

    return fields


def extract_label_info(object_file_path):

    """
    Creates a dictionary of k-v pairs where each key is a span ID and
    each value is a label. Allows us to determine what the label is
    for any "position" in the text, because you can use the label dict
    to lookup the span_id.
    """

    # This is NER, so each label can have one or several token IDs associated with it.
    # Each line/list `l` holds the token IDs from index 2 up to the hash symbol.
    with open(object_file_path) as t:
        labs = [l.split() for l in t.readlines()]
        ids_labs_only = [(l[2: l.index('#')], l[1]) for l in labs]

    # `seen` keeps track of span_ids we have already added as key-label pairs.
    # Some label files have redundant tags (spans have multiple labels),
    # so we need to only include the first instance of each span.
    # Open book_194.objects for an example of this; span 26217, 'Украины',
    # is tagged both as part of an Org AND on its own as LocOrg, even though
    # it only occurs once in the sentence.
    ind_dicts = []
    seen = []
    for item in ids_labs_only:
        for span_id in item[0]:
            if span_id not in seen:
                ind_dicts.append({span_id: item[1]})
                seen.append(span_id)

    # Following the for loop, `label_dict` is now a dict with span_ids as keys
    # and their labels as values.
    label_dict = {}
    for d in ind_dicts:
        label_dict.update(d)

    return label_dict


def create_span_df(extracted_span_data, label_dict):

    """
    Takes the extracted span data from 2 steps back (list of lists), and
    the label dict from the last step, and creates the span + label DF for
    joining with the token DF."""

    # Create span DF from the extracted span data
    span_df = pd.DataFrame(extracted_span_data).iloc[:, 1:]
    span_df.columns = ['position_id', 'mention_type', 'begin_index', 'n_tokens']

    # Convert dtypes for joining with token df
    for col in ['position_id', 'n_tokens', 'begin_index']:
        span_df[col] = span_df[col].astype('int')

    # Assign labels to each position
    get_label = lambda x: label_dict.get(str(x), np.nan)
    span_df['label'] = span_df['position_id'].apply(get_label)

    return span_df


def create_final_label_df(token_df, span_df):

    """
    create_final_label_df joins the token_df and span_df together,
    then applies nested conditional logic to assign the correct prefix
    to each label.

    The possible situations:
      1. (cond1) A single token has a label and the word before it is not labeled
      2. (cond2) A token is labeled, and is the first word of a multi-word phrase
      3. (cond3) A token is labeled, and is a non-first word of a multi-word phrase

    (1) and (2) correspond to a 'B-' prefix, and (3) corresponds to the 'I-' prefix.
    These are standard prefixes used in NER to help the model know which tokens
    are the "B"eginning of a labeled phrase and which are "I"nside the phrase.

    This ends up looking like B-ORG (organization), I-PER (person), etc.
    The four possible labels in this dataset are Org, Location, Person, and LocOrg.
    """

    label_df = token_df.merge(span_df,
                              left_on='begin_index',
                              right_on='begin_index',
                              how='left')

    # Position IDs being associated with multiple mention types created
    # duplicate tokens in the join, so we drop those now
    label_df.drop_duplicates(subset=['begin_index', 'text'],
                             keep='first', inplace=True)

    # SQL-like way of keeping track of the n_tokens value from prev row
    label_df['lag(n_tokens)'] = label_df['n_tokens'].shift(1)

    cond1 = (label_df['n_tokens'] == 1) & (label_df['lag(n_tokens)'].isna())
    label_df['label1'] = np.where(cond1, 'B-' + label_df['label'], np.nan)

    cond2 = (label_df['n_tokens'] >= 1) & (label_df['lag(n_tokens)'].isna())
    label_df['label2'] = np.where(cond2, 'B-' + label_df['label'], np.nan)

    # Label and lag(n_tokens) are both non-null (indicates being inside of a phrase)
    cond3 = (label_df['label'].notna()) & (label_df['lag(n_tokens)'].notna())
    label_df['label3'] = np.where(cond3, 'I-' + label_df['label'], np.nan)

    # If 'label1' is not null, use 'label1', etc., ... if 'label3' is null, use 'label'
    # ('label' will be NaN for non-labeled words, which is what we want)
    label_df['final_label'] = \
        np.where(label_df['label1'].notna(),
                 label_df['label1'],
                 np.where(label_df['label2'].notna(),
                          label_df['label2'],
                         np.where(label_df['label3'].notna(),
                                  label_df['label3'],
                                 label_df['label'])))

    # Fill null labels with 'O' (conventional)
    label_df['final_label'] = label_df['final_label'].fillna('O')

    # Return only text and labels
    return label_df[['text', 'final_label']]


def process_and_save_example(dataset, prefix):

    # Process a single set of files corresponding to one train/test example
    # Provide either 'devset' or 'testset' to dataset arg

    token_file = f'../data/ru/factRuEval-2016/{dataset}/{prefix}.tokens'
    span_file = f'../data/ru/factRuEval-2016/{dataset}/{prefix}.spans'
    extracted = extract_span_info(span_file)
    label_file = f'../data/ru/factRuEval-2016/{dataset}/{prefix}.objects'
    save_path = f'../data/ru/factRuEval-2016/{dataset}/clean/{prefix}.txt'

    token_df = create_token_df(token_file)
    extracted_span_data = extract_span_info(span_file)
    label_dict = extract_label_info(label_file)
    span_df = create_span_df(extracted_span_data, label_dict)
    final_df = create_final_label_df(token_df, span_df)
    final_df.to_csv(save_path, sep=' ', header=False, index=False,
                    quoting=csv.QUOTE_NONE)

    return None

def join_all_examples(data_path, output_filename):

    """
    Joins all the examples in a folder into one text file.
    This allows us to easily join with other datasets later.

    Spaces between sentences were also lost during earlier
    processing steps, so we add them back in here.
    """

    filenames = os.listdir(data_path)

    with open(f'{data_path}/{output_filename}', 'w') as outfile:
        for fname in filenames:
            with open(f'{data_path}/{fname}') as infile:
                outfile.write(infile.read())

    return None


def tidy_output(joined_output_filepath):

    """Takes the joined data and does some final touches for cleanliness.

    A few labels were not caught by the conditional logic applied earlier,
    but it's few enough that going back and rewriting the cond. logic would
    not be a good use of time, considering this data only needs to be
    processed once. Instead, we tidy it up here using regex."""

    with open(joined_output_filepath, 'r') as outfile:
        o = outfile.readlines()

    # Remove blank rows that were left in
    o = [line for line in o if line != ' O\n']
    o_new = []
    o_final = []
    ins_label_re = re.compile('\s(I-[A-Z][a-z]+)')
    beg_label_re = re.compile('\s(B-[A-Z][a-z]+)')

    # Fix labeling: if a label starts with I and prev line lacks an identical B- label,
    # change the I to B. Similarly, if a label starts with B but the prev
    # line has a B, the current line should be changed to an I.

    # We need a new list (o_final) to apply the second rule (conflicts arise
    # if trying to do it all during one iteration loop).

    for line in o:

        if ins_label_re.search(line) \
        and not ins_label_re.search(line).groups(0)[0].replace('I-', 'B-') \
        in o[o.index(line) - 1]:
            label = ins_label_re.search(line).groups(0)[0]
            fixed_label = label.replace('I-', 'B-')
            fixed_line = line.replace(label, fixed_label)
            o_new.append(fixed_line)

        else:
            o_new.append(line)

        # Add newlines between sentences
        if line == '. O\n':
            o_new.append('\n')

    # Second pass for second condition
    for line in o_new:

        if beg_label_re.search(line) and beg_label_re.search(line).groups(0)[0] \
                                            in o_new[o_new.index(line) - 1]:
            label = beg_label_re.search(line).groups(0)[0]
            fixed_label = label.replace('B-', 'I-')
            fixed_line = line.replace(label, fixed_label)
            o_final.append(fixed_line)

        else:
            o_final.append(line)

    with open(joined_output_filepath, 'w') as outfile:
        outfile.write(''.join(o_final))

    return None


def main():

    """
    Complete processing of factRuEval-2016.
    Reads in every single example in each dataset (devset, testset),
    cleans them, appends them with labels, joins them together,
    performs some final sanity checks, and outputs the final product
    into one file per dataset.
    """

    datasets = ['devset', 'testset']
    input_paths = ['../data/ru/factRuEval-2016/devset',
                   '../data/ru/factRuEval-2016/testset']
    output_paths = ['../data/ru/factRuEval-2016/devset/clean',
                   '../data/ru/factRuEval-2016/testset/clean']
    output_files = ['devset_joined.txt', 'testset_joined.txt']

    # Process inputs
    for dataset, path in zip(datasets, input_paths):
        filenames = pd.Series(sorted(os.listdir(path)), name='name')

        # Get an array of all the prefixes 'book_*' (unique example identifiers)
        prefixes = filenames.str.extract('(book_\d+)') \
                            .drop_duplicates().dropna().values.reshape(-1)

        for prefix in prefixes:
            process_and_save_example(dataset, prefix)

    # Join and tidy final output files (1 output file for each dataset)
    for path, file in zip(output_paths, output_files):
        join_all_examples(path, file)
        joined_output_filepath = f'{path}/{file}'
        tidy_output(joined_output_filepath)


if __name__ == '__main__':
    main()
