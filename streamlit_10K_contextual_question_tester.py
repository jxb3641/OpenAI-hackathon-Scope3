import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import requests
from OpenAIUtils import get_embedding, call_openai_api_completion, produce_prompt, questions_to_answers, file_to_embeddings
from EDGARFilingUtils import (
    get_all_submission_ids, 
    get_text_from_files_for_submission_id, 
    split_text, 
    filter_chunks, 
    ROOT_DATA_DIR,
    TICKER_TO_COMPANY_NAME,
    QUESTION_TO_CATEGORY
)
import openai
openai.api_key = st.secrets["openai_api_key"]

from transformers import GPT2TokenizerFast
# Used to check the length of the prompt.
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

st.set_page_config(layout="wide")

### Streamlit app starts here
st.title("Play with GPT-3 Completion API and 10-Ks")

list_of_questions = QUESTION_TO_CATEGORY.keys()

relevant_questions = st.multiselect("Select questions to use for search within the text.",
                                    list_of_questions,default=list_of_questions)
full_file_path = datadir / f"{file_name}.txt"
re_embed = not st.checkbox("Re-calculate Document Embeddings")
if st.button("Search for relevant sections to list of questions"):
    textChunks = filter_chunks(split_text(text))
    st.write(f"{len(textChunks)} Filtered chunks parsed.")
    st.write("Retrieving Embeddings...")
    embeddings = file_to_embeddings(full_file_path,text_chunks=textChunks,use_cache=re_embed)
    st.write("Embeddings Retrieved.")
    answers = questions_to_answers(relevant_questions,embeddings)
    st.subheader("Top results")
    st.table(pd.concat(answers))