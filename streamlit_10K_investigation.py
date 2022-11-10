import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import requests
import openai
openai.api_key = st.secrets["openai_api_key"]

from EDGARFilingUtils import get_random_sample_filings, does_text_have_climate_keywords, concat_keyword_sentences
from OpenAIUtils import call_openai_api_completion

st.set_page_config(layout="wide")


### Streamlit app starts here
st.title("Toying around with GPT-3 and 10-Ks")

df_10k_text = get_random_sample_filings(seed=1)

filing = st.selectbox("Select Filing",list(df_10k_text.index))

item1, item1metrics, mda, mdametrics = st.columns(4)

with item1:
    st.subheader("Item 1 text")
    text = df_10k_text.loc[filing,'item1_txt'].replace(". ",". \n").replace("$","\$")
    keyword_sentences, keyword_counts = does_text_have_climate_keywords(text)
    components.html(f'<span style="word-wrap:break-word;">{text}</span>', height=800,scrolling=True)
    #question = st.text_area("Ask GPT-3 about Item1",placeholder="Enter a prompt...")
    #response = call_openai_api_completion(text[:500],question)
    #components.html(f'<span style="word-wrap:break-word;">{response}</span>', height=200,scrolling=True)


with item1metrics:
    st.subheader(f"Keyword Hits ({sum(keyword_counts.values())} total)")
    text = df_10k_text.loc[filing,'item1_txt'].replace(". ",". \n").replace("$","\$")
    keyword_sentences, keyword_counts = does_text_have_climate_keywords(text)
    st.table(pd.DataFrame.from_dict(keyword_counts,orient="index"))
    st.subheader("Sentences by keyword")
    st.json(json.dumps(keyword_sentences),expanded=False)
    question = st.text_area("Ask GPT-3 about Item1 sentences containing keywords",placeholder="Enter a prompt...")
    response = call_openai_api_completion(concat_keyword_sentences(keyword_sentences),question)
    components.html(f'<span style="word-wrap:break-word;">{response}</span>', height=200,scrolling=True)
    
with mda:
    st.subheader("MDA text")
    text = df_10k_text.loc[filing,'mda_txt'].replace(". ",". \n").replace("$","\$")
    keyword_sentences, keyword_counts = does_text_have_climate_keywords(text)
    components.html(f'<span style="word-wrap:break-word;">{text}</span>', height=800,scrolling=True)
    #question = st.text_area("Ask GPT-3 about MDA",placeholder="Enter a prompt...")
    #response = call_openai_api_completion(text[:500],question)
    #components.html(f'<span style="word-wrap:break-word;">{response}</span>', height=200,scrolling=True)

with mdametrics:
    st.subheader(f"Keyword Hits ({sum(keyword_counts.values())} total)")
    text = df_10k_text.loc[filing,'item1_txt'].replace(". ",". \n").replace("$","\$")
    keyword_sentences, keyword_counts = does_text_have_climate_keywords(text)
    st.table(pd.DataFrame.from_dict(keyword_counts,orient="index"))
    st.subheader("Sentences by keyword")
    st.json(json.dumps(keyword_sentences),expanded=False)
    question = st.text_area("Ask GPT-3 about MDA sentences containing keywords",placeholder="Enter a prompt...")
    response = call_openai_api_completion(concat_keyword_sentences(keyword_sentences),question)
    components.html(f'<span style="word-wrap:break-word;">{response}</span>', height=200,scrolling=True)
