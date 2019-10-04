import numpy as np
import os
import re
import spacy
from spacy.gold import Doc, biluo_tags_from_offsets

# Processes factRuEval-2016 dataset from separate files containing text,
# objects, spans, tokens, coreferences, etc. into one file per dataset
# (devset, testset), with each line being a token-BILUO label pair, and
# sequences separated by newlines. Only uses the .txt and .spans data files.

# Recently added: Shared Task 2019 data is now appended to the devset and
# testset produced by factRu!

FACTRU_LABEL_DICT = {
    "facility_descr": "MISC",
    "geo_adj": "LOC",
    "job": "MISC",
    "loc_descr": "LOC",
    "loc_name": "LOC",
    "name": "PER",
    "nickname": "PER",
    "org_descr": "ORG",
    "org_name": "ORG",
    "patronymic": "PER",
    "prj_name": "MISC",
    "prj_descr": "MISC",
    "surname": "PER",
}


def open_file(path, mode="r", form="string"):
    # Convenience function
    with open(path, mode) as f:
        if form == "string":
            return f.read()
        else:
            return f.readlines()


def process_pair(pair, dataset_dir, label_dict):

    """
    Inputs:
    pair: (___.txt, ___.spans) tuple containing the filenames for each example.
    dataset_dir: str: which dataset directory the files live in.

    Outputs:
    formatted_lines: string containing the processed and formatted tokens and their
    corresponding labels.
    """

    pair_paths = os.path.join(dataset_dir, pair[0]), os.path.join(dataset_dir, pair[1])
    txt, spans = open_file(pair_paths[0]), open_file(pair_paths[1], form="lines")

    # Extract the tag type, index, end index (index + length), and entity
    span_lists = [l.split() for l in spans]
    span_tups = [(int(i[2]), int(i[2]) + int(i[3]), i[1]) for i in span_lists]

    # Convert the text to a spacy Doc (for compatibility with `biluo_tags_from_offsets`)
    nlp = spacy.load("xx_ent_wiki_sm")
    doc = nlp(txt, disable=["ner"])

    # Create the token-label pairs using `biluo_tags_from_offsets`
    tokens_biluo = list(zip(doc.doc, biluo_tags_from_offsets(doc, span_tups)))

    # Remove label prefixes and standardize label names (see LABEL_DICT at top)
    # `tokens_biluo` is a list of tuples, and tuples are immutable, so we need
    # to use a workaround
    tokens_biluo_temp = []
    for tup in tokens_biluo:
        if tup[1] != "O" and tup[1][2:] != "":
            new_lab = label_dict[tup[1][2:]]
            tokens_biluo_temp.append((tup[0], new_lab))
        else:
            tokens_biluo_temp.append((tup[0], tup[1]))

    # Spacy's tokenization is space-preserving, and this will cause
    # problems with the BERT model, so we replace those with standard newlines
    tokens_biluo = [
        tup if str(tup[0]).strip() != "" else "\n" for tup in tokens_biluo_temp
    ]

    # Format lines for writing out
    formatted_lines = ["\t".join(str(s) for s in tup) + "\n" for tup in tokens_biluo]
    for i, line in enumerate(formatted_lines):
        if line == ".\tO\n":
            formatted_lines.insert(i + 1, "\n\n")

    return formatted_lines


def process_and_save_factRu(factRu_path, combined_path, label_dict):

    """
    Inputs: factRu_path, the path to factRuEval-2016.

    Outputs: No direct outputs; the processed data is written to
    the combined data directory, to be appended to by the shared task data.
    """

    for dataset in ["devset", "testset"]:

        dataset_dir = os.path.join(factRu_path, dataset)
        dataset_dir_list = os.listdir(dataset_dir)

        all_spans = sorted([fn for fn in dataset_dir_list if ".spans" in fn])
        all_txt = sorted([fn[:-6] + ".txt" for fn in all_spans])
        pairs = zip(all_txt, all_spans)

        output_filename = os.path.join(combined_path, f"{dataset}_combined.txt")
        with open(output_filename, "w") as outfile:
            for pair in pairs:
                outfile.writelines(process_pair(pair, dataset_dir, label_dict))

        print(f"Wrote all processed examples in {dataset} to {output_filename}.")


def prep_st_data(raw_file, ann_file):

    # Article title begins at line 4
    with open(raw_file) as f:
        raw = "".join(f.readlines()[4:])

    # Skip first line (filename)
    with open(ann_file) as f:
        objs = f.readlines()[1:]

    return raw, objs


def find_exact_matches(raw, objs):

    """
    Uses `re` to find exact matches in the Shared Task data, then creates
    a list of tuples `ents`, where each tuple is the starting index, ending index,
    and label of each entity.

    For context while reading this implementation, ST's data looks like this:
    Асия Биби	Асия Биби	PER	PER-Asia-Bibi
    Высокий суд в Лахоре	Высокий суд в Лахоре	ORG	ORG-Supreme-Court-of-Pakistan
    Корана	Коран	PRO	PRO-The-Quran
    Лахоре	Лахор	LOC	GPE-Lahore

    (Note that the first half of each line is of varying length, and sometimes
    contains two different spellings. Thankfully, any redundant tagging created by
    1-letter-off-spellings will be handled later by spaCy's biluo_tags_from_offsets.)
    """

    ents = []
    for line in objs:
        spl = line.split()
        # The last 2 positions in spl are label data (see docstring)
        phrase_len_t = len(spl) - 2
        phrase_len = int(phrase_len_t / 2)
        # Separate the first and alternate spelling of each phrase
        phrase, phrase_alt = " ".join(spl[:phrase_len]), " ".join(spl[phrase_len:-2])
        # If alternate spelling is listed, search for both
        if phrase != phrase_alt:
            for p in [phrase, phrase_alt]:
                match = re.search(p, raw)
                if match:
                    ents.append((match.start(), match.end(), spl[-2]))
        else:
            match = re.search(phrase, raw)
            if match:
                ents.append((match.start(), match.end(), spl[-2]))

    return ents


def process_pair_ST(prefix):

    """
    Similar to process_pair for the factRuEval data, but with tweaks for
    the Shared Task 2019 data.
    """

    raw_path = f"../data/ru/shared_task_2019/raw/{prefix}.txt"
    ann_path = f"../data/ru/shared_task_2019/annotated/{prefix}.out"

    raw, objs = prep_st_data(raw_path, ann_path)
    ents = find_exact_matches(raw, objs)

    # Convert the text to a spacy Doc (for compatibility with `biluo_tags_from_offsets`)
    nlp = spacy.load("xx_ent_wiki_sm")
    doc = nlp(raw, disable=["ner"])

    # Create the token-label pairs using `biluo_tags_from_offsets`
    tokens_biluo = list(zip(doc.doc, biluo_tags_from_offsets(doc, ents)))

    # Remove prefixes ("B-", "I-", etc.) from labels
    # `tokens_biluo` is a list of tuples, and tuples are immutable, so we need
    # to use a workaround
    tokens_biluo_temp = []
    for tup in tokens_biluo:
        if tup[1] != "O":
            new_lab = tup[1][2:]
            tokens_biluo_temp.append((tup[0], new_lab))
        else:
            tokens_biluo_temp.append((tup[0], tup[1]))

    # Spacy's tokenization is space-preserving, and this will cause
    # problems with the BERT model, so we replace those with standard newlines
    tokens_biluo = [
        tup if str(tup[0]).strip() != "" else "\n" for tup in tokens_biluo_temp
    ]

    # Format lines for writing out:
    # Insert newlines to separate each sentence
    # Remove any leftover space artifacts from spacy processing
    formatted_lines = ["\t".join(str(s) for s in tup) + "\n" for tup in tokens_biluo]
    for i, line in enumerate(formatted_lines):
        if line == ".\tO\n":
            formatted_lines.insert(i + 1, "\n\n")
        elif line[0].isspace() and line != "\n\n":
            formatted_lines.remove(line)

    return formatted_lines


def process_and_append_ST(shared_task_path, combined_path):

    """
    Does the preprocessing for ST data and appends it to the
    already-processed factRu datasets.
    """

    prefixes = [
        pref[:-4]
        for pref in sorted(os.listdir(os.path.join(shared_task_path, "raw")))
        if pref != ".DS_Store"
    ]

    processed_data_list = []
    for prefix in prefixes:
        processed_data_list.append(process_pair_ST(prefix))

    # Randomly permute the processed data before appending to dev and test sets
    np.random.seed(6)
    n = len(processed_data_list)
    train_prop = int(n * 0.85)
    np.random.shuffle(processed_data_list)  # shuffle is an inplace operation
    train_data = processed_data_list[:train_prop]
    test_data = processed_data_list[train_prop:]

    for data, outfile in zip([train_data, test_data], ["devset", "testset"]):
        with open(os.path.join(combined_path, f"{outfile}_combined.txt"), "a") as f:
            for lines in data:
                f.writelines(lines)

    print("Finished processing and appending the Shared Task data.")


def cleanup(combined_path):

    """
    Resulting data has an inconsistent number of newlines between
    each example. This fn performs a quick cleanup to make sure
    later processing results in the correct # of examples.
    """

    for data_name in ["devset", "testset"]:
        fp = os.path.join(combined_path, f"{data_name}_combined.txt")
        with open(fp, "r") as f:
            txt = f.read()
            replace_n = lambda txt, n: txt.replace("\n" * n, "\n\n")
            for n in [3, 4, 5]:
                txt = replace_n(txt, n)
        with open(fp, "w") as f:
            f.write(txt)

        n_examples = len(txt.split("\n\n"))
        print(f"Finished data tidying. {data_name.title()} has {n_examples} examples.")
