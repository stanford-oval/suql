"""
Functionality to work with .prompt files
"""

from typing import List
import openai
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List
import openai
from openai import OpenAIError
from functools import partial
from datetime import date
from jinja2 import Environment, FileSystemLoader, select_autoescape
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)

#singleton
jinja_environment = Environment(loader=FileSystemLoader('./'),
                  autoescape=select_autoescape(), trim_blocks=True, lstrip_blocks=True, line_comment_prefix='#')
# uncomment if using Azure OpenAI
openai.api_base = 'https://ovalopenairesource.openai.azure.com/'
openai.api_type = 'azure'
openai.api_version = '2022-12-01' # this may change in the future


inference_cost_per_1000_tokens = {'ada': 0.0004, 'babbage': 0.0005, 'curie': 0.002, 'davinci': 0.02, 'turbo': 0.002} # for Azure
total_cost = 0 # in USD

def get_total_cost():
    global total_cost
    return total_cost

def _model_name_to_cost(model_name: str) -> float:
    for model_family in inference_cost_per_1000_tokens.keys():
        if model_family in model_name:
            return inference_cost_per_1000_tokens[model_family]
    raise ValueError('Did not recognize GPT-3 model name %s' % model_name)

@retry(retry=retry_if_exception_type(OpenAIError), wait=wait_random_exponential(multiplier=0.5, max=20), stop=stop_after_attempt(6))
def openai_completion_with_backoff(**kwargs):
    global total_cost
    ret =  openai.Completion.create(**kwargs)
    total_tokens = ret['usage']['total_tokens']
    total_cost += total_tokens / 1000 * _model_name_to_cost(kwargs['engine'])
    
    return ret

def _fill_template(template_file, prompt_parameter_values):
    template = jinja_environment.get_template(template_file)

    # always make these useful constants available in a template
    today = date.today() # make a new function call each time since the date might change during a long-term server deployment
    prompt_parameter_values['today'] = today
    prompt_parameter_values['current_year'] = today.year
    prompt_parameter_values['chatbot_name'] = 'RestaurantGenie'

    filled_prompt = template.render(**prompt_parameter_values)
    filled_prompt = '\n'.join([line.strip() for line in filled_prompt.split('\n')]) # remove whitespace at the beginning and end of each line
    return filled_prompt

def _generate(filled_prompt, engine, max_tokens, temperature, stop_tokens, top_p, frequency_penalty, presence_penalty, postprocess, max_tries, ban_line_break_start):
    # don't try multiple times if the temperature is 0, because the results will be the same
    if max_tries > 1 and temperature == 0:
        max_tries = 1

    logger.info('LLM input = %s', filled_prompt)

    # ChatGPT has specific input tokens. The following is a naive implementation of few-shot prompting, which may be improved.
    if engine == "gpt-35-turbo":
        filled_prompt = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % filled_prompt
        ban_line_break_start = False # no need to prevent new lines in ChatGPT
        if stop_tokens is None:
            stop_tokens = []
        stop_tokens.append('<|im_end|>')
    
    for _ in range(max_tries):
        no_line_break_start = ''
        no_line_break_length = 0
        if ban_line_break_start:
            no_line_break_length = 3
            # generate 3 tokens that definitely are not line_breaks
            no_line_break_start = openai_completion_with_backoff(engine=engine,
                                                        prompt=filled_prompt,
                                                        max_tokens=no_line_break_length,
                                                        temperature=temperature,
                                                        top_p=top_p,
                                                        frequency_penalty=frequency_penalty,
                                                        presence_penalty=presence_penalty,
                                                        stop=stop_tokens,
                                                        logit_bias={'198': -100, '628': -100} # \n, \n\n
                                                        )['choices'][0]['text']
            
        generation_output = openai_completion_with_backoff(engine=engine,
                                                        prompt=filled_prompt + no_line_break_start,
                                                        max_tokens=max_tokens - no_line_break_length,
                                                        temperature=temperature,
                                                        top_p=top_p,
                                                        frequency_penalty=frequency_penalty,
                                                        presence_penalty=presence_penalty,
                                                        stop=stop_tokens,
                                                        )
        generation_output = no_line_break_start + generation_output['choices'][0]['text']
        logger.info('LLM output = %s', generation_output)

        generation_output = generation_output.strip()
        if postprocess:
            generation_output = _postprocess_generations(generation_output)

        if len(generation_output) > 0:
            break


    return generation_output

def _postprocess_generations(generation_output: str) -> str:
    """
    Might output an empty string if generation is not at least one full sentence
    """
    # replace all whitespaces with a single space
    generation_output = ' '.join(generation_output.split())

    # remove extra dialog turns, if any
    turn_indicators = ['You:', 'They:', 'Context:', 'You said:', 'They said:', 'Assistant:', 'Chatbot:', "User:"]
    for t in turn_indicators:
        if generation_output.find(t) > 0:
            generation_output = generation_output[:generation_output.find(t)]

    generation_output = generation_output.strip()
    # delete half sentences
    if len(generation_output) == 0:
        return generation_output

    if generation_output[-1] not in {'.', '!', '?'}:
        last_sentence_end = max(generation_output.find(
            '.'), generation_output.find('!'), generation_output.find('?'))
        if last_sentence_end > 0:
            generation_output = generation_output[:last_sentence_end+1]

    return generation_output

def llm_generate(template_file: str, prompt_parameter_values: dict, engine,
            max_tokens, temperature, stop_tokens, top_p=0.9, frequency_penalty=0, presence_penalty=0,
            postprocess=True, max_tries=1, ban_line_break_start=False, filled_prompt=None):
    """
    filled_prompt gives direct access to the underlying model, without having to load a prompt template from a .prompt file. Used for testing.
    ban_line_break_start can potentially double the cost, though in practice (and especially with good prompts) this only happens for a fraction of inputs
    """
    if filled_prompt is None:
        filled_prompt = _fill_template(template_file, prompt_parameter_values)
    return _generate(filled_prompt, engine, max_tokens, temperature, stop_tokens, top_p, frequency_penalty, presence_penalty, postprocess, max_tries, ban_line_break_start)
    

def batch_llm_generate(template_file: str, prompt_parameter_values: List[dict], engine,
            max_tokens, temperature, stop_tokens, top_p=0.9, frequency_penalty=0, presence_penalty=0,
            postprocess=True, max_tries=1, ban_line_break_start=False, max_num_threads=10):
    """
    We use multithreading here (instead of multiprocessing) because this method is I/O-bound, mostly waiting for an HTTP response to come back.
    """

    f = partial(_generate, engine=engine,
            max_tokens=max_tokens, temperature=temperature, stop_tokens=stop_tokens, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
            postprocess=postprocess, max_tries=max_tries, ban_line_break_start=ban_line_break_start)

    with ThreadPoolExecutor(max_num_threads) as executor:
        thread_outputs = [executor.submit(f, _fill_template(template_file, p)) for p in prompt_parameter_values]
    thread_outputs = [o.result() for o in thread_outputs]
    return thread_outputs