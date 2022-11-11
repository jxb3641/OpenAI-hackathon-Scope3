import openai
from openai.embeddings_utils import cosine_similarity
import numpy as np
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# For the purposes of measuring relevance between long docs and short queries,
# We want to use the text-search doc and query embeddings 
# (https://beta.openai.com/docs/guides/embeddings/what-are-embeddings)
# Keep these here so that we can simply refer to the model family when using these wrappers.
EMBEDDING_MODELS = {"ada": {"query": "text-search-ada-query-001",
                            "doc": "text-search-ada-doc-001",
                            "completion": "text-ada-001"
                            },
                    "babbage": {"query": "text-search-babbage-query-001",
                                "doc": "text-search-babbage-doc-001",
                                "completion": "text-babbage-001"
                                },
                    "curie": {"query": "text-search-curie-query-001",
                              "doc": "text-search-curie-doc-001",
                              "completion": "text-curie-001"
                              },
                    "davinci": {"query": "text-search-curie-query-001",
                                "doc": "text-search-curie-doc-001",
                                "completion": "text-davinci-002"
                                }
                    }

def call_openai_api_completion(prompt, model_family='ada'):
    """Send a request to OpenAI's text generation API endpoint,
    with send_prompt and model.

    Args:
        prompt (str): The full prompt. 
        model_family (str, optional): model family to use for generation. Can be any of "ada", "babbage", "curie", "davinci". 
        Defaults to 'ada'.

    Returns:
        str: The top scoring autocompletion. 
    """

    response = openai.Completion.create(
      model=EMBEDDING_MODELS[model_family]["completion"],
      prompt=prompt,
      max_tokens=50,
      temperature=0.2,
    )
    return response['choices'][0]['text']

def get_embedding(text, model_family="babbage"):
    """Given a string of long-form text, produce the embedding using the corresponding text-search-doc API endpoint.

    Args:
        text (str): String to produce an embedding for.
        model_family (str, optional): OpenAI model family to use text-search-doc for. Can be any of "ada", "babbage", "curie", "davinci".
        Defaults to "babbage".

    Returns:
        np.ndarray: Vector representation of the text.
    """

    # All embeddings can only embed up to 2024 token strings.
    if tokenizer(text) > 2020:
         return np.ndarray([]) # Return an empty array.
    embedding = openai.Embedding.create(input = [text], model=EMBEDDING_MODELS[model_family]["doc"])['data'][0]['embedding']
    return embedding

def query_similarity_search(embeddings, query, model_family="babbage", n=3, pprint=True):
    """Search the doc embeddings for the most similar matches with the query.

    Args:
        embeddings (DataFrame): df containing 'text' field, and its search/doc embeddings.
        query (str): Question to embed.  Uses the 'query' version of the embedding model.
        model_family (str, optional): model name.  can be "davinci", "curie", "babbage", "ada"; Default "babbage"
        n (int, optional): number of top results. Defaults to 3.
        pprint (bool, optional): Whether to print the text and scores of the top results. Defaults to True.

    Returns:
       DataFrame: Top n rows of the embeddings DataFrame, with similarity column added. Sorted by similarity score from highest to lowest. 
    """
    embedded = get_embedding(query,EMBEDDING_MODELS[model_family]['query'])
    embeddings["similarities"] = embeddings["doc_embeddings"].apply(lambda x: cosine_similarity(x, embedded))

    res = embeddings.sort_values("similarities", ascending=False).head(n)
    if pprint:
        for _, series in res.iterrows():
            print(series["similarities"],series["text"])
            print()
    return res

def produce_prompt(context, query_text):
    """Produce the prompt by appending the query text with the context.

    Args:
        context (str): Context to try to answer the question with.
        query_text (str): Question to ask.

    Returns:
        str: Prompt to prime GPT-3 completion API endpoint.
    """

    return f"""Given Context:\n\n{context}\n\n{query_text}"""

#TODO: Add caching/saving functionality to the embeddings calls (or maybe these should be done separately in scripts that use these functions?)
