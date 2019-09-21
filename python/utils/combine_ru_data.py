import os
from spacy import load
from spacy.gold import Doc, biluo_tags_from_offsets

# Processes factRuEval-2016 dataset from separate files containing text,
# objects, spans, tokens, coreferences, etc. into one file per dataset
# (devset, testset), with each line being a token-BILUO label pair, and
# sequences separated by newlines. Only uses the .txt and .spans data files.


def open_file(path, mode="r", form="string"):
    # Convenience function
    with open(path, mode) as f:
        if form == "string":
            return f.read()
        else:
            return f.readlines()


def process_pair(pair, dataset_dir):

    """
    Inputs:
    pair: (___.txt, ___.spans) tuple containing the filenames for each example.
    dataset_dir: str: which dataset directory the files live in.

    Outputs:
    formatted_lines: string containing the processed and formatted tokens and their corresponding
    BILUO tags.
    """

    pair_paths = os.path.join(dataset_dir, pair[0]), os.path.join(dataset_dir, pair[1])
    txt, spans = open_file(pair_paths[0]), open_file(pair_paths[1], form="lines")

    # Extract the tag type, index, end index (index + length), and entity
    span_lists = [l.split() for l in spans]
    span_tups = [(int(i[2]), int(i[2]) + int(i[3]), i[1]) for i in span_lists]

    # Convert the text to a spacy Doc (for compatibility with `biluo_tags_from_offsets`)
    nlp = load("xx_ent_wiki_sm")
    doc = nlp(txt, disable=["ner"])

    # Create the token-label pairs using `biluo_tags_from_offsets`
    tokens_biluo = list(zip(doc.doc, biluo_tags_from_offsets(doc, span_tups)))

    # Spacy's tokenization is space-preserving, and this will cause
    # problems with the BERT model, so we replace those with standard newlines
    tokens_biluo = [tup if str(tup[0]).strip() != "" else "\n" for tup in tokens_biluo]

    # Format lines for writing out
    formatted_lines = ["\t".join(str(s) for s in tup) + "\n" for tup in tokens_biluo]

    return formatted_lines


def process_and_save_data(ru_data_path):

    """
    Inputs: ru_data_path, the path to factRuEval-2016.
    (Fn is not compatible with other datasets.)

    Outputs: No direct outputs; the processed data is written to
    the same directory as the original dataset.
    """

    for dataset in ["devset", "testset"]:

        dataset_dir = os.path.join(ru_data_path, dataset)
        dataset_dir_list = os.listdir(dataset_dir)

        all_spans = sorted([fn for fn in dataset_dir_list if ".spans" in fn])
        all_txt = sorted([fn[:-6] + ".txt" for fn in all_spans])
        pairs = zip(all_txt, all_spans)

        output_filename = os.path.join(ru_data_path, dataset, f"{dataset}_combined.txt")
        with open(output_filename, "w") as outfile:
            for pair in pairs:
                outfile.writelines(process_pair(pair, dataset_dir))

        print(f"Wrote all processed examples in {dataset} to {output_filename}.")


if __name__ == "__main__":
    ru_data_path = "../../data/ru/factRuEval-2016/"
    process_and_save_data(ru_data_path)
