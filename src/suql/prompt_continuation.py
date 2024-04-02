"""
Functionality to work with .prompt files
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List

import openai
from openai import OpenAI

client = OpenAI()
import os
import time
import traceback
from functools import partial
from threading import Thread

import pymongo
from jinja2 import Environment, FileSystemLoader, select_autoescape

from suql.utils import num_tokens_from_string

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create file handler which logs even debug messages
fh = logging.FileHandler("prompts.log")
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

jinja_environment = Environment(
    loader=FileSystemLoader(os.path.dirname(os.path.abspath(__file__))),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
    line_comment_prefix="#",
)
# # uncomment if using Azure OpenAI
openai.api_type == "open_ai"
# openai.api_type = "azure"
# openai.api_base = "https://ovalopenairesource.openai.azure.com/"
# openai.api_version = "2023-05-15"

mongo_client = pymongo.MongoClient("localhost", 27017)
prompt_cache_db = mongo_client["open_ai_prompts"]["caches"]

# inference_cost_per_1000_tokens = {'ada': 0.0004, 'babbage': 0.0005, 'curie': 0.002, 'davinci': 0.02, 'turbo': 0.003, 'gpt-4': 0.03} # for Azure
inference_input_cost_per_1000_tokens = {
    "gpt-4": 0.03,
    "gpt-3.5-turbo-0613": 0.0010,
    "gpt-3.5-turbo-1106": 0.0010,
    "gpt-4-1106-preview": 0.01,
}  # for OpenAI
inference_output_cost_per_1000_tokens = {
    "gpt-4": 0.06,
    "gpt-3.5-turbo-0613": 0.0010,
    "gpt-3.5-turbo-1106": 0.0020,
    "gpt-4-1106-preview": 0.03,
}  # for OpenAI
total_cost = 0  # in USD


def get_total_cost():
    global total_cost
    return total_cost


def _model_name_to_cost(model_name: str) -> float:
    if (
        model_name in inference_input_cost_per_1000_tokens
        and model_name in inference_output_cost_per_1000_tokens
    ):
        return (
            inference_input_cost_per_1000_tokens[model_name],
            inference_output_cost_per_1000_tokens[model_name],
        )
    raise ValueError("Did not recognize GPT model name %s" % model_name)


def openai_chat_completion_with_backoff(**kwargs):
    global total_cost
    ret = client.chat.completions.create(**kwargs)
    num_prompt_tokens = ret.usage.prompt_tokens
    num_completion_tokens = ret.usage.completion_tokens
    prompt_cost, completion_cost = _model_name_to_cost(kwargs["model"])
    total_cost += (
        num_prompt_tokens / 1000 * prompt_cost
        + num_completion_tokens / 1000 * completion_cost
    )  # TODO: update this
    return ret.choices[0].message.content


def _fill_template(template_file, prompt_parameter_values):
    template = jinja_environment.get_template(template_file)

    filled_prompt = template.render(**prompt_parameter_values)
    filled_prompt = "\n".join(
        [line.strip() for line in filled_prompt.split("\n")]
    )  # remove whitespace at the beginning and end of each line
    return filled_prompt


def _generate(
    filled_prompt,
    engine,
    max_tokens,
    temperature,
    stop_tokens,
    top_p,
    frequency_penalty,
    presence_penalty,
    postprocess,
    max_tries,
    ban_line_break_start,
):
    # don't try multiple times if the temperature is 0, because the results will be the same
    if max_tries > 1 and temperature == 0:
        max_tries = 1

    logger.info("LLM input = %s", filled_prompt)

    for _ in range(max_tries):
        no_line_break_start = ""
        no_line_break_length = 0
        kwargs = {
            "messages": [
                {"role": "system", "content": filled_prompt + no_line_break_start}
            ],
            "max_tokens": max_tokens - no_line_break_length,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop_tokens,
        }
        if openai.api_type == "azure":
            kwargs.update({"engine": engine})
        else:
            engine_model_map = {
                "gpt-4": "gpt-4",
                "gpt-35-turbo": "gpt-3.5-turbo-1106",
                "gpt-3.5-turbo": "gpt-3.5-turbo-1106",
                "gpt-4-turbo": "gpt-4-1106-preview",
            }
            # https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models
            # https://platform.openai.com/docs/models/model-endpoint-compatibility
            kwargs.update(
                {
                    "model": (
                        engine_model_map[engine]
                        if engine in engine_model_map
                        else engine
                    )
                }
            )

        generation_output = openai_chat_completion_with_backoff(**kwargs)
        generation_output = no_line_break_start + generation_output
        logger.info("LLM output = %s", generation_output)

        generation_output = generation_output.strip()
        if postprocess:
            generation_output = _postprocess_generations(generation_output)

        if len(generation_output) > 0:
            break

    logger.info(f"total cost this run: {total_cost}")
    return generation_output


def _postprocess_generations(generation_output: str) -> str:
    """
    Might output an empty string if generation is not at least one full sentence
    """
    # replace all whitespaces with a single space
    generation_output = " ".join(generation_output.split())

    # remove extra dialog turns, if any
    turn_indicators = [
        "You:",
        "They:",
        "Context:",
        "You said:",
        "They said:",
        "Assistant:",
        "Chatbot:",
        "User:",
    ]
    for t in turn_indicators:
        if generation_output.find(t) > 0:
            generation_output = generation_output[: generation_output.find(t)]

    generation_output = generation_output.strip()
    # delete half sentences
    if len(generation_output) == 0:
        return generation_output

    if generation_output[-1] not in {".", "!", "?"}:
        last_sentence_end = max(
            generation_output.find("."),
            generation_output.find("!"),
            generation_output.find("?"),
        )
        if last_sentence_end > 0:
            generation_output = generation_output[: last_sentence_end + 1]

    return generation_output


def call_with_timeout(func, timeout_sec, *args, **kwargs):
    class FunctionThread(Thread):
        def __init__(self, func, *args, **kwargs):
            Thread.__init__(self)
            self.func = func
            self.args = args
            self.kwargs = kwargs
            self.return_value = None
            self.exception = None
            self.traceback_str = None

        def run(self):
            try:
                self.return_value = self.func(*self.args, **self.kwargs)
            except Exception as e:
                self.exception = e
                self.traceback_str = traceback.format_exc()

    func_thread = FunctionThread(func, *args, **kwargs)
    func_thread.start()
    func_thread.join(timeout_sec)

    if func_thread.is_alive():
        print(f"Function timed out after {timeout_sec} seconds.")
        return False, None
    elif func_thread.exception:
        print(f"Function raised an exception: {func_thread.exception}")
        print(func_thread.traceback_str)
        return False, None
    else:
        print("Function finished successfully.")
        return True, func_thread.return_value


def llm_generate(
    template_file: str,
    prompt_parameter_values: dict,
    engine,
    max_tokens,
    temperature,
    stop_tokens,
    top_p=0.9,
    frequency_penalty=0,
    presence_penalty=0,
    postprocess=True,
    max_tries=1,
    ban_line_break_start=False,
    filled_prompt=None,
    attempts=2,
    max_wait_time=None,
):
    """
    filled_prompt gives direct access to the underlying model, without having to load a prompt template from a .prompt file. Used for testing.
    ban_line_break_start can potentially double the cost, though in practice (and especially with good prompts) this only happens for a fraction of inputs
    """
    start_time = time.time()
    if filled_prompt is None:
        filled_prompt = _fill_template(template_file, prompt_parameter_values)

    cache_res = prompt_cache_db.find_one(
        {
            "model": engine,
            "prompt": filled_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop_tokens": stop_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
    )
    if cache_res:
        return cache_res["res"], 0

    # We have experiences very long latency from time to time from both Azure's and OpenAI's chatGPT response time
    # Here is a heuristics-based, dynamically-calculated max wait time, before we cancel the last request and re-issue a new one
    total_token = num_tokens_from_string(filled_prompt) + max_tokens
    if max_wait_time is None:
        max_wait_time = 0.05 * total_token + 1

    success = False
    final_result = None

    for attempt in range(attempts):
        print(f"Attempt {attempt + 1} of {attempts}")
        success, result = call_with_timeout(
            _generate,
            max_wait_time,
            filled_prompt,
            engine,
            max_tokens,
            temperature,
            stop_tokens,
            top_p,
            frequency_penalty,
            presence_penalty,
            postprocess,
            max_tries,
            ban_line_break_start,
        )
        if success:
            final_result = result
            break
        else:
            print("Retrying...")

    if final_result is None:
        final_result = _generate(
            filled_prompt,
            engine,
            max_tokens,
            temperature,
            stop_tokens,
            top_p,
            frequency_penalty,
            presence_penalty,
            postprocess,
            max_tries,
            ban_line_break_start,
        )

    end_time = time.time()
    elapsed_time = end_time - start_time

    prompt_cache_db.insert_one(
        {
            "model": engine,
            "prompt": filled_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop_tokens": stop_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "res": final_result,
        }
    )

    return final_result, elapsed_time


def batch_llm_generate(
    template_file: str,
    prompt_parameter_values: List[dict],
    engine,
    max_tokens,
    temperature,
    stop_tokens,
    top_p=0.9,
    frequency_penalty=0,
    presence_penalty=0,
    postprocess=True,
    max_tries=1,
    ban_line_break_start=False,
    max_num_threads=10,
):
    """
    We use multithreading here (instead of multiprocessing) because this method is I/O-bound, mostly waiting for an HTTP response to come back.
    Currently not used by the SUQL repo, but could be brought back in a future version.
    """

    f = partial(
        _generate,
        engine=engine,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_tokens=stop_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        postprocess=postprocess,
        max_tries=max_tries,
        ban_line_break_start=ban_line_break_start,
    )

    with ThreadPoolExecutor(max_num_threads) as executor:
        thread_outputs = [
            executor.submit(f, _fill_template(template_file, p))
            for p in prompt_parameter_values
        ]
    thread_outputs = [o.result() for o in thread_outputs]
    return thread_outputs
