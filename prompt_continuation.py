"""
GPT-3 continues a prompt, works with any .prompt file
"""

from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List
import openai
from functools import partial
from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)

#singleton
jinja_environment = Environment(loader=FileSystemLoader('./'),
                  autoescape=select_autoescape(), trim_blocks=True, lstrip_blocks=True, line_comment_prefix='#')
# uncomment if using Azure OpenAI
# openai.api_base = 'https://ovalopenairesource.openai.azure.com/'
# openai.api_type = 'azure'
# openai.api_version = '2022-12-01' # this may change in the future


def _fill_template(template_file, prompt_parameter_values):
    template = jinja_environment.get_template(template_file)
    filled_prompt = template.render(**prompt_parameter_values)
    filled_prompt = '\n'.join([line.strip() for line in filled_prompt.split('\n')]) # remove whitespace at the beginning and end of each line
    return filled_prompt

def _generate(filled_prompt, engine, max_tokens, temperature, stop_tokens, top_p, frequency_penalty, presence_penalty, max_tries):
    # don't try multiple times if the temperature is 0, because the results will be the same
    if max_tries > 1 and temperature == 0:
        max_tries = 1

    logger.info('LLM input = %s', filled_prompt)

    for _ in range(max_tries):
        generation_output = openai.Completion.create(engine=engine,
                                                        prompt=filled_prompt,
                                                        max_tokens=max_tokens,
                                                        temperature=temperature,
                                                        top_p=top_p,
                                                        frequency_penalty=frequency_penalty,
                                                        presence_penalty=presence_penalty,
                                                        best_of=1,
                                                        stop=stop_tokens,
                                                        logprobs=0,  # log probability of top tokens
                                                        )
        generation_output = generation_output['choices'][0]['text']
        logger.info('LLM output = %s', generation_output)

        generation_output = generation_output.strip()

        if len(generation_output) > 0:
            break

    return generation_output

def llm_generate(template_file: str, prompt_parameter_values: dict, engine,
            max_tokens, temperature, stop_tokens, top_p=0.9, frequency_penalty=0, presence_penalty=0,
            max_tries=1):

    filled_prompt = _fill_template(template_file, prompt_parameter_values)
    return _generate(filled_prompt, engine, max_tokens, temperature, stop_tokens, top_p, frequency_penalty, presence_penalty, max_tries)
    

def batch_llm_generate(template_file: str, prompt_parameter_values: List[dict], engine,
            max_tokens, temperature, stop_tokens, top_p=0.9, frequency_penalty=0, presence_penalty=0,
            max_tries=1, max_num_threads=10):
    """
    We use multithreading here (instead of multiprocessing) because this method is I/O-bound, mostly waiting for an HTTP response to come back.
    """

    f = partial(_generate, engine=engine,
            max_tokens=max_tokens, temperature=temperature, stop_tokens=stop_tokens, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
            max_tries=max_tries)

    with ThreadPoolExecutor(max_num_threads) as executor:
        thread_outputs = [executor.submit(f, _fill_template(template_file, p)) for p in prompt_parameter_values]
    thread_outputs = [o.result() for o in thread_outputs]
    return thread_outputs