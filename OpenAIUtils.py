import openai
import backoff


def call_openai_api_completion(context,prompt_suffix,model='text-ada-001'):
    """Send a request to OpenAI's text generation API endpoint,
    with send_prompt and model.

    Args:
        context (str): The context. forms the first part of the prompt to send to GPT-3.
        prompt_suffix (str): The rest of the prompt to send for generation.
        model (str, optional): model to use for generation. Defaults to 'text-ada-001'.

    Returns:
        : _description_
    """

    full_prompt = context+f"\n\n{prompt_suffix}"

    response = openai.Completion.create(
      model=model,
      prompt=full_prompt,
      max_tokens=50,
      temperature=0.2,
    )
    return response['choices'][0]['text']


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError),max_tries=15)
def get_embedding(text, model="text-similarity-davinci-001"):
    """Given a string of text, produce the embedding based on the OpenAI model endpoint.

    Args:
        text (str): String to produce an embedding for.
        model (str, optional): OpenAI model endpoint to target. Defaults to "text-similarity-davinci-001".

    Returns:
        np.ndarray: Vector representation of the text.
    """
    embedding = openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
    #print(text)
    #print("#"*50)
    return embedding