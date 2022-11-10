"""Utilities for 10K filings."""
import glob
import re
import random
import json
import pandas as pd


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

    return text.split(". ")

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
