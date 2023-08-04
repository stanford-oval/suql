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
import json

#singleton
jinja_environment = Environment(loader=FileSystemLoader('./'),
                  autoescape=select_autoescape(), trim_blocks=True, lstrip_blocks=True, line_comment_prefix='#')
# uncomment if using Azure OpenAI
openai.api_base = 'https://oval-france-central.openai.azure.com/'
openai.api_type = 'azure'
openai.api_version = '2023-05-15' # this may change in the future


inference_cost_per_1000_tokens = {'ada': 0.0004, 'babbage': 0.0005, 'curie': 0.002, 'davinci': 0.02, 'turbo': 0.002, 'chat': 0.002} # for Azure
total_cost = 0 # in USD

def get_total_cost():
    global total_cost
    return total_cost

def _model_name_to_cost(model_name: str) -> float:
    for model_family in inference_cost_per_1000_tokens.keys():
        if model_family in model_name:
            return inference_cost_per_1000_tokens[model_family]
    raise ValueError('Did not recognize GPT-3 model name %s' % model_name)

# @retry(retry=retry_if_exception_type(OpenAIError), wait=wait_random_exponential(multiplier=0.5, max=20), stop=stop_after_attempt(6))
def openai_chat_completion_with_backoff(**kwargs):
    global total_cost
    ret =  openai.ChatCompletion.create(**kwargs)
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
    # filled_prompt = '\n'.join([line.strip() for line in filled_prompt.split('\n')]) # remove whitespace at the beginning and end of each line
    return filled_prompt

def _generate(filled_prompt_system, filled_prompt_user, engine, max_tokens, temperature, stop_tokens, top_p, frequency_penalty, presence_penalty, max_tries, log_file_path):    
    with open(log_file_path, "a") as fd:
    
        # don't try multiple times if the temperature is 0, because the results will be the same
        if max_tries > 1 and temperature == 0:
            max_tries = 1

        messages = []
        if filled_prompt_system != "": 
            messages.append({"role": "system", "content": filled_prompt_system})
        if filled_prompt_user != "":
            messages.append({"role": "user", "content": filled_prompt_user})
        
        
        if filled_prompt_system == "":
            fd.write('LLM input (user) = {}\n'.format(filled_prompt_user))
        elif filled_prompt_user == "":
            fd.write('LLM input (system) = {}\n'.format(filled_prompt_system))
        else:
            fd.write('LLM input = {}\n'.format(json.dumps(messages, indent=2)))
        
        for _ in range(max_tries):
            no_line_break_length = 0
                
            generation_output = openai_chat_completion_with_backoff(engine=engine,
                                                            messages=messages,
                                                            max_tokens=max_tokens - no_line_break_length,
                                                            temperature=temperature,
                                                            top_p=top_p,
                                                            frequency_penalty=frequency_penalty,
                                                            presence_penalty=presence_penalty,
                                                            stop=stop_tokens,
                                                            )
            generation_output = generation_output['choices'][0]['message']['content']

            generation_output = generation_output.strip()

            if len(generation_output) > 0:
                break
        
        fd.write('LLM output = {}\n'.format(generation_output))

    return generation_output


def llm_generate(template_file_system: str, template_file_user: str, prompt_parameter_values: dict, engine,
            max_tokens, temperature, stop_tokens, top_p=0.9, frequency_penalty=0, presence_penalty=0, max_tries=1, all_user=False, all_system=False, log_file_path="prompts.log"):
    """
    filled_prompt gives direct access to the underlying model, without having to load a prompt template from a .prompt file. Used for testing.
    ban_line_break_start can potentially double the cost, though in practice (and especially with good prompts) this only happens for a fraction of inputs
    """
    if all_user:
        filled_prompt = _fill_template(template_file_user, prompt_parameter_values)
        return _generate("", filled_prompt, engine, max_tokens, temperature, stop_tokens, top_p, frequency_penalty, presence_penalty, max_tries, log_file_path)
    elif all_system:
        filled_prompt = _fill_template(template_file_user, prompt_parameter_values)
        return _generate(filled_prompt, "", engine, max_tokens, temperature, stop_tokens, top_p, frequency_penalty, presence_penalty, max_tries, log_file_path)
    else:
        filled_prompt_system = _fill_template(template_file_system, prompt_parameter_values)
        filled_prompt_user = _fill_template(template_file_user, prompt_parameter_values)
        return _generate(filled_prompt_system, filled_prompt_user, engine, max_tokens, temperature, stop_tokens, top_p, frequency_penalty, presence_penalty, max_tries, log_file_path)
