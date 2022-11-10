# OpenAI Hackathon for the Climate 2022 : Scope3

## Introduction
This project utilizes OpenAI's LLMs to create our awesome hackathon idea.

## Overview

### data/ directory
This directory contains the data being read in.  As of this writing, this includes:
- `10K/2021_q1.zip`: A zipped directory of 2021 quarter 1 10-Ks from 8000 different companies. When setting up, this should be unzipped into the folder
`10K/q1/` for the EDGARFilingUtils functions to work properly.
For each Submission ID there are assumed to be three `.txt` files:
- The entire parsed 10-K filing, converted into text after HTML/XBRL tags have been replaced with newlines.
- The Item1 section of this filing, where newlines and markdown tags have been removed.
- The Item7/MDA section of this filing, where newlines and markdown tags have have been removed.

### Convenience function modules

#### EDGARFilingUtils.py
Contains convenience functions for loading in 10-K filings data described above, splitting text, concatenating text fragments and finding text containing any hits of a pre-defined list of climate keywords in the text. 

#### OpenAIUtils.py
Contains wrapper functions for calling the OpenAI API.  Currently contains wrappers for the completion and embedding API endpoints.

#### streamlit_10K_investigation.py
An interactive Streamlit application to interface with the 10-K text in Item1 and MDA sections. 
After setting up the project, use `streamlit run streamlit_10K_investigation.py`, and the app will open in your local browser. 

## Setup

This project was built off Python 3.8.4. 
First, clone this repository: 

`git clone https://github.com/jxb3641/OpenAI-hackathon-Scope3.git`

In a terminal:
- change directories into the cloned repo
- Create a new virtual environment: `python3 -m venv .venv`
- Activate the virtual environment: `source .venv/bin/activate`.  You know you did it right if you see the activated virtual environment in your terminal prompt, like (.venv) user@workstation: ~/OpenAI-hackathon-Scope3
- Install the `requirements.txt`: `pip install -r requirements.txt` 
- To interface with OpenAI, you need to set up an API key. 
    - Go to openai.com.
    - Click on the 'API' tab on the top right.
    - Click on "Login" button at the top right.
    - Sign up with the email address you provided to OpenAI when signing up for this hackathon.  You should see "oai-hackathon-2022-team-8" as one of your organizations.
    - Click on your initial on the top right. A dropdown menu should expand.
    - Click on "View API Keys" -> "Create a new secret key"
    - Copy this key.
    - Create a `.env` file in the root directory of this repo. 
    - In the `.env` file, create a new key, value pair: `OPENAI_API_KEY=(CopiedKeyHere)`. Save and exit.
    - Whenever you want to use the OpenAI API in your scripts , be sure to include this line so your client knows your key: 
    `openai.api_key=os.getenv("OPENAI_API_KEY")`

To setup streamlit to see your API key, you will need to create a `secrets.toml` file in the `.streamlit` directory at root of this project.
For the `streamlit_10k_investigation.py` streamlit app, you simply need to add the following line to the `secrets.toml`:
`openai_api_key = "(yourAPIKeyHere)"`

