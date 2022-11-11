import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import requests
from OpenAIUtils import get_embedding, call_openai_api_completion, produce_prompt
from EDGARFilingUtils import get_all_submission_ids, get_text_from_files_for_submission_id
import openai
openai.api_key = st.secrets["openai_api_key"]

from transformers import GPT2TokenizerFast
# Used to check the length of the prompt.
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

st.set_page_config(layout="wide")

### Streamlit app starts here
st.title("Play with GPT-3 Completion API with 10-Ks")


with st.sidebar:
    submission_id = st.selectbox("10-K Submission ID",options=get_all_submission_ids())
    file_to_use = st.radio("Which text file do you want to use?",
                            options=("full","item1","mda"),
                            index=0,horizontal=True)
    text = get_text_from_files_for_submission_id(submission_id)[f"{file_to_use}_txt"]
    text = text.replace("$","\$")
    st.subheader("GPT-3 Params")
    model_family = st.radio("Select model family.",  help="ada is cheapest and fastest. Davinci is strongest, but more expensive and slower.",
                     options=("ada","babbage","curie","davinci"),
                     index=2,
                     horizontal=True)
    model_temp = st.number_input("Completion Temperature",
                                  help = "0 leads to more deterministic, consistent answers. 1 leads to less predictable answers.",
                                  value=0.,
                                  min_value=0.,
                                  max_value=1.,
                                  step=0.05 
                                )
    context = st.text_area("Context",placeholder="Paste relevant snippets in here.  Must not be longer than 2048 tokens, or 4000 tokens for davinci.")
    question = st.text_area("Question",placeholder="Enter a question...")

    gpt3_prompt = produce_prompt(context,question)
    prompt_token_size = len(tokenizer.encode(gpt3_prompt)) 
    st.write(f"Total prompt token length: {prompt_token_size}")
    if model_family == "davinci":
        # davinci can take longer total prompts.
        if prompt_token_size > 4000:
            st.error("Total prompt is too long.  Try cutting down the context.")
    else:
        if prompt_token_size  > 2048:
            st.error("Total prompt is too long.  Try cutting down the context.")

    gpt3_response = ""
    if st.button("Run Completion"):
        gpt3_response = call_openai_api_completion(gpt3_prompt,model_family)


    st.subheader("GPT-3 Response")
    components.html(f'<span style="word-wrap:break-word;">{gpt3_response}</span>', height=800,scrolling=True)

st.subheader(f"{file_to_use} Text")
st.markdown(text)