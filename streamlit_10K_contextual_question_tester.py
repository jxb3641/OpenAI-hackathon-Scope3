import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import requests
from OpenAIUtils import get_embedding, call_openai_api_completion, produce_prompt, questions_to_answers, file_to_embeddings
from EDGARFilingUtils import get_all_submission_ids, get_text_from_files_for_submission_id, split_text, filter_chunks, ROOT_DATA_DIR
import openai
openai.api_key = st.secrets["openai_api_key"]

from transformers import GPT2TokenizerFast
# Used to check the length of the prompt.
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

st.set_page_config(layout="wide")

### Streamlit app starts here
st.title("Play with GPT-3 Completion API and 10-Ks")

#TODO: Christina/Simon add in list of relevant questions to search through 10-Ks
list_of_questions = ["Climate change?"]

industry_category_subdir = {"Food Beverage":"4_food_bev","Transportation":"11_transportation"}

with st.sidebar:
    industry_category = st.selectbox("Select the industry of interest.",options=["Food Beverage","Transportation"])
    data_source = st.selectbox("Select the data source.",options=["10k"])

    if data_source == "10k":
        file_to_use = st.radio("Which file type do you want to use?",
                                options=("full","item1","mda"),
                                index=0,horizontal=True)
    elif data_source == "transcripts":
        file_to_use = st.radio("Which file type do you want to use?",
                                options=(".csv",".html"),
                                index=0,horizontal=True)

    datadir = ROOT_DATA_DIR / industry_category_subdir[industry_category] / data_source
    file_name = st.selectbox("10-K Filename",options=get_all_submission_ids(datadir=datadir))
    text = get_text_from_files_for_submission_id(file_name,datadir=datadir)[f"{file_to_use}_txt"]
    st.subheader("GPT-3 Params")
    model_family = st.radio("Select model family.",  help="ada is cheapest and fastest. Davinci is strongest, but more expensive and slower.",
                     options=("ada","babbage","curie","davinci"),
                     index=3,
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
        gpt3_response = call_openai_api_completion(gpt3_prompt,model_family,temperature=model_temp)


    st.subheader("GPT-3 Response")
    components.html(f'<span style="word-wrap:break-word;">{gpt3_response}</span>', height=800,scrolling=True)
full_text_tab, search_tab = st.tabs([f"Filtered {file_to_use} text", "Search Results"])

with full_text_tab:
    filtered_text = "\n\n".join(filter_chunks(split_text(text)))
    st.markdown(filtered_text)

with search_tab:
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