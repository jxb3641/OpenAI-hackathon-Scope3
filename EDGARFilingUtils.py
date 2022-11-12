"""Utilities for 10K filings."""
import glob
import re
import random
import json
import pandas as pd


#TODO: Refactor this into two functions: 
# one that takes in a submission id, and creates a dict of item1/mda sources, and texts
# another that takes in the directory, and outputs all submission ids
# random sample filings should be done separately, or in a third function.

def get_all_submission_ids(datadir="data/10K/q1"):
    """get all the submission IDs of 10-K .txts.  
    Assumes filing texts are of form (submission-id).txt, (submission-id)_item1.txt, (submission-id)_mda.txt

    Args:
        datadir (str): Where to look for the text files.
    Returns:
        (tuple(str)): Tuple of unique submission IDs.
    """

    tenk_all_filingnames = set([re.search("\d+-\d+-\d+",fp).group() for fp in glob.glob(f"{datadir}/*.txt")])

    return tuple(tenk_all_filingnames)

def get_text_from_files_for_submission_id(submission_id, datadir="data/10K/q1"):
    """Read in the .txt files for submission_id, located in datadir. 

    Args:
        submission_id (str): Submission id of the filing.
        datadir (str): filepath where all 3 files (.txt, item1.txt, mda.txt) for the submission id should be located.  

    Returns:
        dict: Dictionary containing the submission id, filepath of the .txt, item1.txt and mda.txt, files, 
        and their texts read in as strings with keys full_txt, item1_txt, mda_txt.
    """

    # Helper function to read in file texts as strings
    def get_text(fp):
        with open(fp, encoding='utf-8') as f:
            text = f.read()
        return text
    text_dict = {}

    for fp in glob.glob(f"{datadir}/{submission_id}*.txt"):
        if re.search("item1.txt",fp):
            text_dict["item1"] = fp
            text_dict["item1_txt"] = get_text(fp)
        elif re.search("mda.txt",fp):
            text_dict["mda"] = fp
            text_dict["mda_txt"] = get_text(fp)
        else:
            text_dict["fullFiling"] = fp
            text_dict["full_txt"] = get_text(fp)


    return text_dict


def get_random_sample_filings(number_filings=50,seed=None):
    """For a random sample of filings, parse their names, MDA, and Item1 .txt files
    and their text.

    Args:
        seed (int, optional): Seed for random instance. Defaults to None.
        number_filings (int, optional): Number of filings to get the MDA, Item1 and full .txt files. Defaults to 50.

    Returns:
        pandas.DataFrame: DF of filing names, the filepaths of the Full, MDA and Item1 text, and their parsed in text. 
    """

    # Helper function to read in file texts as strings
    def get_text(fp):
        with open(fp) as f:
            text = f.read()
        return text

    random_inst = random.Random(seed) if seed else random.Random()

    # All .txt files in the data directory have name {digits}-{digits}-{digits} as their prefix, and 
    # one of _mda.txt, _item1.txt, or just .txt as suffixes. The RE below just captures the common prefixes.
    tenk_all_filingnames = [re.search("\d+-\d+-\d+",fp).group() for fp in glob.glob("data/10K/q1/*.txt")]

    # Pull number_filings (fullFiling, MDA, and Item1) filename triples
    txt_by_filing = {}
    for filing_num in random_inst.sample(tenk_all_filingnames,number_filings):
        txt_by_filing[filing_num] = {}
        for fp in glob.glob(f"data/10K/q1/{filing_num}*.txt"): # Find the 3 files with the common filing prefix
            if re.search("item1.txt",fp):
                txt_by_filing[filing_num]["item1"] = fp
            elif re.search("mda.txt",fp):
                txt_by_filing[filing_num]["mda"] = fp
            else:
                txt_by_filing[filing_num]["fullFiling"] = fp

    # DF indexed by filing prefix, with columns "item1", "mda", "fullFiling".  
    # Add in 3 more columns to contain the text strings.
    df = pd.read_json(json.dumps(txt_by_filing),orient='index')
    for col in df.columns:
        df[col+"_txt"] = df[col].apply(lambda x: get_text(x))
    return df

def split_text(text):
    """split text into workable chunks.

    Args:
        text (str): original text.

    Returns:
        list(str): list of text chunks.
    """

    #TODO: Filter out table of contents, anything past item 15

    return text.split("\n\n")


def filter_text(split_text):
    """Filter text"""

    filtered_split = [] 
    #Remove chunks less than some 
    for chunk in split_text:
        if len(chunk)<20:
            continue
        filtered_split.append(chunk)

    return split_text
  

def does_text_have_climate_keywords(text):
    """Checks if any of a preset list of keywords is in the text.

    Args:
        text (str): text to search for keywords.
    Returns:
        A dict of sentences featuring keywords, a dict of keyword counts in the text.
    """

    keywords = [
        "energy",
        "electric vehicle",
        "climate change",
        "wind (power|energy)",
        "greenhouse gas",
        "solar",
        "\bair\b",
        "carbon",
        "emission",
        "extreme weather",
        "carbon dioxide",
        "battery",
        "pollution",
        "environment",
        "clean power",
        "onshore",
        "coastal area",
        "footprint",
        "charge station",
        "eco friendly",
        "sustainability",
        "energy reform",
        "renewable",
    ]

    keyword_contexts = {keyword : [] for keyword in keywords}
    keyword_counts = {keyword : 0 for keyword in keywords}

    # pre-process text
    split_text = text.lower().split(". ")

    # Count occurrences for each keyword in the text.
    for keyword in keywords:
        for sentence in split_text:
            if re.search(keyword,sentence):
                keyword_contexts[keyword].append(sentence)
                keyword_counts[keyword] = keyword_counts[keyword] + len(re.findall(keyword,sentence))
    return keyword_contexts, keyword_counts

def concat_keyword_sentences(keyword_sentence_map,max_str_length=900):
    """Take in a dictionary of keyword to sentences, and concatenate them up to max_str_length.

    Args:
        keyword_sentence_map (dict): dictionary of sentences by keyword.
        max_str_length (int, optional): maximum length of the concated string. Defaults to 900.

    Returns:
        str: concatenated string of keyword sentences, of length approximately max_str_length characters.
    """

    keyword_sentence_list = [ sent for sentlist in keyword_sentence_map.values() for sent in sentlist]
    concat_str = ""
    while len(concat_str)<max_str_length:
        for keyword_sentence in keyword_sentence_list:
            concat_str += keyword_sentence+"\n\n" 
    return concat_str
